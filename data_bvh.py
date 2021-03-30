import torch
import numpy as np


class BvhDataset(torch.utils.data.Dataset):
    def __init__(self, bvh_file):
        'Initialization'
        self.bvh_file = bvh_file
        self.bvh_frame = np.loadtxt(bvh_file, dtype=np.float32)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.bvh_frame) - 1

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load current frame and next frame
        X = self.bvh_frame[index]
        y = self.bvh_frame[index + 1]

        return X, y
