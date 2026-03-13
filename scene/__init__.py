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
import torch
import numpy as np
import yaml
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from torch.utils.data import DataLoader
from scene.dataloader import *


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, dataset_type='rfid', load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.batch_size = 1
        self.dataset_type = dataset_type
        # cameras_extent controls world-space pruning threshold (0.1 * extent)
        # and clone/split scaling threshold (percent_dense * extent).
        self.cameras_extent = 2

        if dataset_type == 'rfid':
            self.datadir = "./data_test200"
            with open(os.path.join(self.datadir, 'gateway_info.yml'), 'r') as f:
                data = yaml.safe_load(f)
            self.r_o = torch.tensor(data['gateway1']['position'], dtype=torch.float32)
            self.gateway_orientation = np.array(data['gateway1']['orientation'])

        elif dataset_type == 'ble':
            self.datadir = "data/Second_room/5GHz/BLE"
            with open(os.path.join(self.datadir, 'gateway_position.yml'), 'r') as f:
                data = yaml.safe_load(f)
            self.r_o = torch.tensor(list(data.values()), dtype=torch.float32)

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        train_index = os.path.join(self.datadir, "train_index.txt")
        test_index  = os.path.join(self.datadir, "test_index.txt")

        if dataset_type == 'rfid':
            if not os.path.exists(train_index) or not os.path.exists(test_index):
                split_dataset(self.datadir, ratio=0.8, dataset_type='rfid')
            self.train_set = Spectrum_dataset(self.datadir, train_index)
            self.test_set  = Spectrum_dataset(self.datadir, test_index)

        elif dataset_type == 'ble':
            # Per-gateway 75/25 split on valid (RSSI != -100) samples.
            # Index files store (tx_idx, gw_idx) pairs; re-generate if missing or stale.
            needs_split = (not os.path.exists(train_index) or
                           not os.path.exists(test_index) or
                           np.loadtxt(train_index, dtype=int).ndim == 1)
            if needs_split:
                split_dataset(self.datadir, ratio=0.8, dataset_type='ble')
            self.train_set = BLE_dataset(self.datadir, train_index)
            self.test_set  = BLE_dataset(self.datadir, test_index)

            # Reinitialize Gaussian positions inside the actual (normalized) room
            # bounds.  Default gaussian_init uses randn*20 centered at 0, which
            # puts >99% of points outside [0,1] normalized space → CUDA overflow.
            all_pos = torch.cat([
                self.train_set.inputs[:, :3],   # TX positions (normalized)
                self.train_set.inputs[:, 3:6],  # GW positions (normalized)
            ], dim=0)
            pos_min = all_pos.min(dim=0).values.cuda()
            pos_max = all_pos.max(dim=0).values.cuda()
            margin  = (pos_max - pos_min) * 0.05

            N = gaussians.get_xyz.shape[0]
            with torch.no_grad():
                from simple_knn._C import distCUDA2
                new_xyz = (torch.rand((N, 3), device="cuda") *
                           (pos_max - pos_min + 2 * margin) +
                           (pos_min - margin))
                gaussians._xyz.data = new_xyz
                dist2 = torch.clamp_min(distCUDA2(new_xyz), 1e-7)
                gaussians._scaling.data = (
                    torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3))
                gaussians.max_radii2D = torch.zeros(N, device="cuda")
            print(f"[BLE] Gaussians reinitialized in normalized bounds "
                  f"[{pos_min.cpu().numpy().round(3)}, "
                  f"{pos_max.cpu().numpy().round(3)}]")


        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

        print(f"[{dataset_type.upper()}] Train size: {len(self.train_set)} | Test size: {len(self.test_set)}")


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def dataset_init(self):
        self.train_iter_dataset = iter(self.train_iter)
        self.test_iter_dataset = iter(self.test_iter)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
