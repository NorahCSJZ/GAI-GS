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
from scene.dataloader import Spectrum_dataset, split_dataset


class Scene:

    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel,
                 dataset_type='rfid', load_iteration=None,
                 shuffle=True, resolution_scales=[1.0]):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.batch_size = 1
        self.dataset_type = dataset_type
        self._epoch = 0
        self.cameras_extent = 2

        if dataset_type != 'rfid':
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        self.datadir = "./RFID/s23"
        with open(os.path.join(self.datadir, 'gateway_info.yml'), 'r') as f:
            data = yaml.safe_load(f)
        self.r_o = torch.tensor(data['gateway1']['position'], dtype=torch.float32)
        self.gateway_orientation = np.array(data['gateway1']['orientation'])

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        train_index = os.path.join(self.datadir, "train_index.txt")
        test_index = os.path.join(self.datadir, "test_index.txt")

        if not os.path.exists(train_index) or not os.path.exists(test_index):
            split_dataset(self.datadir, ratio=0.8, dataset_type='rfid')
        self.train_set = Spectrum_dataset(self.datadir, train_index)
        self.test_set = Spectrum_dataset(self.datadir, test_index)

        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size,
                                     shuffle=True, num_workers=0)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size,
                                    shuffle=False, num_workers=0)

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
