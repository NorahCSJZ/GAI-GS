#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def split_dataset(datadir, ratio=0.1, dataset_type='rfid'):
    """Randomly split the RFID spectrum dataset into train and test subsets."""
    if dataset_type != "rfid":
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

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
    """RFID spectrum dataset."""

    def __init__(self, datadir, indexdir) -> None:
        super().__init__()
        self.datadir = datadir
        self.tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')
        self.spectrum_dir = os.path.join(datadir, 'spectrum')
        self.dataset_index = np.loadtxt(indexdir, dtype=str)
        self.tx_pos = pd.read_csv(self.tx_pos_dir).values
        self.n_samples = len(self.dataset_index)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        img_name = os.path.join(self.spectrum_dir, self.dataset_index[index] + '.png')
        spectrum = imageio.imread(img_name) / 255.0
        spectrum = torch.tensor(spectrum, dtype=torch.float32)

        tx_pos_i = torch.tensor(
            self.tx_pos[int(self.dataset_index[index]) - 1],
            dtype=torch.float32,
        )

        return tx_pos_i, spectrum


dataset_dict = {"rfid": Spectrum_dataset}
