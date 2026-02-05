#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import datetime
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import DataLoader, DistributedSampler
from scene.dataloader import *
import yaml
import torch

class CollateFunction:
    """Picklable collate function class for DDP multiprocessing"""
    def __init__(self, data_type):
        self.data_type = data_type
    
    def __call__(self, batch):
        """Stack individual samples into batch data"""
        tx_positions, spectra = zip(*batch)
        # Stack spectrum: [B, H, W] for RFID
        spectra_batch = torch.stack(spectra, dim=0)
        # Stack tx_pos: [B, 3]
        tx_positions_batch = torch.stack(tx_positions, dim=0)
        
        # Validate BLE data type shape
        if self.data_type == 'ble':
            if tx_positions_batch.dim() != 2 or tx_positions_batch.shape[1] != 4:
                sample_info = f"First sample shape: {tx_positions[0].shape if len(tx_positions) > 0 else 'N/A'}, value: {tx_positions[0] if len(tx_positions) > 0 else 'N/A'}"
                raise ValueError(
                    f"BLE data type error: tx_positions shape should be [B, 4], actual: {tx_positions_batch.shape}.\n"
                    f"  {sample_info}\n"
                    f"  Please check if BLE_dataset.__getitem__ returns inputs with shape [4]"
                )
            if spectra_batch.dim() != 2 or spectra_batch.shape[1] != 1:
                sample_info = f"First sample shape: {spectra[0].shape if len(spectra) > 0 else 'N/A'}, value: {spectra[0] if len(spectra) > 0 else 'N/A'}"
                raise ValueError(
                    f"BLE data type error: spectra shape should be [B, 1], actual: {spectra_batch.shape}.\n"
                    f"  {sample_info}\n"
                    f"  Please check if BLE_dataset.__getitem__ returns labels with shape [1]"
                )
        
        # Unified return order: (tx_positions_batch, spectra_batch)
        return tx_positions_batch, spectra_batch


def directions_to_quaternion(r_d, eps=1e-6):
    """
    Convert direction vector r_d (n_rays, 3) to quaternion (n_rays, 4)
    Assumes default forward direction is +Z axis (0, 0, 1)
    
    Args:
        r_d: torch.Tensor, shape [n_rays, 3], each unit direction vector
    Returns:
        quat: torch.Tensor, shape [n_rays, 4], quaternion (x, y, z, w)
    """
    device = r_d.device
    dtype = r_d.dtype

    # Normalize input direction
    r_d = r_d / (r_d.norm(dim=-1, keepdim=True) + eps)

    # Default forward direction
    default_forward = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).unsqueeze(0)  # [1,3]

    # Rotation axis = cross(default, target)
    axis = torch.cross(default_forward.expand_as(r_d), r_d, dim=-1)
    axis_norm = axis.norm(dim=-1, keepdim=True)
    parallel_mask = axis_norm < eps
    axis = axis / (axis_norm + eps)

    # Rotation angle = arccos(dot(default, target))
    dot = (default_forward * r_d).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(dot)

    # Quaternion q = [axis * sin(theta/2), cos(theta/2)]
    half = angle / 2
    sin_half = torch.sin(half)
    quat = torch.cat([axis * sin_half, torch.cos(half)], dim=-1)  # [n_rays, 4]

    # For directions close to parallel, directly set as unit quaternion
    quat[parallel_mask.squeeze(-1)] = torch.tensor([0, 0, 0, 1], dtype=dtype, device=device)

    return quat

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, data_type='rfid', load_iteration=None, shuffle=True, resolution_scales=[1.0], ddp=False, rank=0, world_size=1):
        """
        :param path: Path to colmap scene main folder.
        """
        self.ddp = ddp
        self.rank = rank
        self.world_size = world_size
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        # RFID batch size
        self.batch_size = 32
        xyz = self.gaussians.get_xyz.detach()
        self.cameras_extent = 0.5 * (xyz.max(dim=0).values - xyz.min(dim=0).values).max().item()
        self.data_type = 'rfid'
        self.datadir = "data_test200"
        
        yaml_file_path = os.path.join(self.datadir, 'gateway_info.yml')
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
        self.r_o = torch.tensor([data['gateway1']['position']], dtype=torch.float32)
        self.gateway_orientation = torch.tensor([data['gateway1']['orientation']], dtype=torch.float32)
  

        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        dataset = dataset_dict[self.data_type]
        train_index = os.path.join(self.datadir, "train_index.txt")
        test_index = os.path.join(self.datadir, "test_index.txt")
       
        if not os.path.exists(train_index) or not os.path.exists(test_index):
            split_dataset(self.datadir, ratio=0.8, dataset_type="rfid")
        
        self.train_set = dataset(self.datadir, train_index)
        self.test_set = dataset(self.datadir, test_index)
        print(f"Dataset loaded, train size: {len(self.train_set)}, test size: {len(self.test_set)}")

        # Use picklable collate function class
        collate_fn = CollateFunction('rfid')
        
        # Reduce num_workers in DDP mode to avoid resource exhaustion
        num_workers = 0 if ddp else 4

        # DDP: use DistributedSampler
        if ddp:
            self.train_sampler = DistributedSampler(
                self.train_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            self.train_iter = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                sampler=self.train_sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )
        else:
            self.train_sampler = None
            self.train_iter = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )
        
        self.test_iter = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def dataset_init(self, epoch=10):
        # DDP: set epoch to ensure different data order each epoch
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.train_iter_dataset = iter(self.train_iter)
        self.test_iter_dataset = iter(self.test_iter)
        
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

