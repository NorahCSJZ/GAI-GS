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
from einops import rearrange


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

    elif dataset_type == "ble":
        rssi = pd.read_csv(os.path.join(datadir, 'gateway_rssi.csv')).values
        num_gateways = rssi.shape[1]
        train_pairs, test_pairs = [], []
        for gw_idx in range(num_gateways):
            valid_tx = np.where(rssi[:, gw_idx] != -100)[0].tolist()
            random.shuffle(valid_tx)
            split = int(len(valid_tx) * ratio)
            for tx_i in valid_tx[:split]:
                train_pairs.append((tx_i, gw_idx))
            for tx_i in valid_tx[split:]:
                test_pairs.append((tx_i, gw_idx))
        np.savetxt(os.path.join(datadir, "train_index.txt"), train_pairs, fmt='%d')
        np.savetxt(os.path.join(datadir, "test_index.txt"),  test_pairs,  fmt='%d')


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


def rssi2amplitude(rssi):
    """Convert RSSI (dBm, negative) to normalized amplitude [0, 1]."""
    return 1 - (rssi / -100)


def amplitude2rssi(amplitude):
    """Convert normalized amplitude [0, 1] back to RSSI (dBm)."""
    return -100 * (1 - amplitude)


class BLE_dataset(Dataset):
    """BLE RSSI dataset.

    Each sample is a (TX position, gateway) pair with a valid RSSI measurement
    (RSSI != -100).  Coordinates are normalized by scale_worldsize so the scene
    fits in roughly [0, 1] on each axis.

    Inputs:  [tx_x, tx_y, tx_z, gw_x, gw_y, gw_z]  normalized, shape [6]
    Labels:  rssi amplitude in [0, 1]                shape [1]

    Expected data layout under datadir/:
        tx_pos.csv            — TX positions, shape [num_tx, 3]
        gateway_rssi.csv      — RSSI table,   shape [num_tx, num_gw]
        gateway_position.yml  — gateway positions as a YAML dict of lists
        train_index.txt / test_index.txt  — (tx_idx, gw_idx) pairs, one per line
    """

    def __init__(self, datadir, indexdir):
        super().__init__()
        self.scale_worldsize = 1
        tx_pos = torch.tensor(
            pd.read_csv(os.path.join(datadir, 'tx_pos.csv')).values,
            dtype=torch.float32)                                    # [num_tx, 3]
        rssis = torch.tensor(
            pd.read_csv(os.path.join(datadir, 'gateway_rssi.csv')).values,
            dtype=torch.float32)                                    # [num_tx, num_gw]
        with open(os.path.join(datadir, 'gateway_position.yml'), 'r') as f:
            gw_data = yaml.safe_load(f)
            gw_pos = torch.tensor([pos for pos in gw_data.values()], dtype=torch.float32)

        pairs = np.loadtxt(indexdir, dtype=int).reshape(-1, 2)
        tx_idx = pairs[:, 0]
        gw_idx = pairs[:, 1]

        # Normalize coordinates so the scene fits in ~[0, 1] on each axis.
        tx_norm = tx_pos / self.scale_worldsize
        gw_norm = gw_pos / self.scale_worldsize

        self.inputs = torch.cat([tx_norm[tx_idx], gw_norm[gw_idx]], dim=1)  # [N, 6]
        self.labels = rssi2amplitude(rssis[tx_idx, gw_idx]).unsqueeze(1)     # [N, 1]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


dataset_dict = {"rfid": Spectrum_dataset, "ble": BLE_dataset}
