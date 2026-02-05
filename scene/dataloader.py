#! /usr/bin/env python
# -*- coding: utf-8 -*-


import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm

def rssi2amplitude(rssi):
    """convert rssi to amplitude
    """
    return 1 - (rssi / -100)


def rssi2amplitude(rssi):
    """convert rssi to amplitude
    """
    return 1 - (rssi / -100)


def amplitude2rssi(amplitude):
    """convert amplitude to rssi
    """
    return -100 * (1 - amplitude)


def euclidian2spherical(x, y, z):
    # Calculate theta
    theta = torch.arctan2(torch.sqrt(x**2 + y**2), z)
    # Calculate phi
    phi = torch.arctan2(y, x)

    return theta, phi

def spherical2euclidian(theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack([x, y, z]).T


def split_dataset(datadir, ratio=0.1, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrum')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')


class Spectrum_dataset(Dataset):
    """Spectrum dataset class."""
    
    def __init__(self, datadir, indexdir) -> None:
        super().__init__()
        self.datadir = datadir  
        self.tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')  
        self.spectrum_dir = os.path.join(datadir, 'spectrum')  
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])       
        self.dataset_index = np.loadtxt(indexdir, dtype=str)  
        self.tx_pos = pd.read_csv(self.tx_pos_dir).values  
        self.n_samples = len(self.dataset_index)  

    def __len__(self):
        return self.n_samples 

    def __getitem__(self, index):
        
        img_name = os.path.join(self.spectrum_dir, self.dataset_index[index] + '.png')
        spectrum = imageio.imread(img_name) / 255.0  
        spectrum = torch.tensor(spectrum, dtype=torch.float32)  

        tx_pos_i = torch.tensor(self.tx_pos[int(self.dataset_index[index]) - 1], dtype=torch.float32)

        return tx_pos_i, spectrum  


class BLE_dataset(Dataset):
    """BLE dataset class."""
    
    def __init__(self, datadir, indexdir) -> None:
        super().__init__()
        self.datadir = datadir  
        self.tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')  
        self.rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')       
        self.rssis = torch.tensor(pd.read_csv(self.rssi_dir).values, dtype=torch.float32)
        self.dataset_index = np.loadtxt(indexdir, dtype=int)  
        self.tx_pos = torch.tensor(pd.read_csv(self.tx_pos_dir).values, dtype=torch.float32) 
        self.inputs, self.labels = self.load_data()
        assert self.inputs.ndim == 2 and self.inputs.shape[1] == 4, \
            f"inputs shape should be [N,4], got {self.inputs.shape}"

    def __len__(self):
        rssis = self.rssis[self.dataset_index]
        return int((rssis != -100).sum().item()) 

    def __getitem__(self, index):

        return self.inputs[index], self.labels[index]  
    
    def load_data(self):
        inputs, labels = [], []

        for idx in tqdm(self.dataset_index, total=len(self.dataset_index)):
            rssis = self.rssis[idx]         # [num_gateways]
            tx    = self.tx_pos[idx]        # [3]
            for i_gateway, rssi in enumerate(rssis):
                if rssi != -100:
                    # [tx_x, tx_y, tx_z, gateway_id]
                    inputs.append(torch.tensor([tx[0], tx[1], tx[2], float(i_gateway)], dtype=torch.float32))
                    labels.append(torch.tensor([rssi], dtype=torch.float32))

        nn_inputs = torch.stack(inputs, dim=0)      # [N, 4]
        nn_labels = torch.stack(labels, dim=0)      # [N, 1]
        nn_labels = rssi2amplitude(nn_labels)       # If needed
        return nn_inputs, nn_labels



dataset_dict = {"rfid": Spectrum_dataset, "ble": BLE_dataset}
