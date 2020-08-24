"""
Class for managing UCF data.
"""
import os.path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class UCFDataset(Dataset):
    """UCF dataset."""

    def __init__(self, csv_file, root_dir='/data', train=True, opt_flow_len=10, transform=None):
        self.data = pd.read_csv(csv_file, usecols=range(3), names=['split', 'label', 'folder'])
        if train:
            self.data = self.data[self.data.split == 'train']
        else:
            self.data = self.data[self.data.split == 'test']
        self.root_dir = root_dir
        self.opt_flow_len = opt_flow_len
        self.transform = transform

        # Useful things
        self.labels = sorted(self.data.label.unique().tolist())
        self.original_image_shape = (341, 256)
        self.image_shape = (224, 224)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            idx = [idx]

        X_spatial_batch = []
        X_temporal_batch = []
        y_batch = []

        for row in self.data.iloc[idx, :].itertuples(index=False):
            # Get the stacked optical flows from disk.
            X_spatial, X_temporal = self.get_static_frame_and_stacked_opt_flows(row)

            # Get the corresponding labels
            y = to_categorical(self.labels.index(row.label), len(self.labels))
            y = np.squeeze(y)

            X_spatial_batch.append(X_spatial)
            X_temporal_batch.append(X_temporal)
            y_batch.append(y)

        sample = {'flow': X_temporal_batch, 'rgb': X_spatial_batch, 'y': y_batch}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_static_frame_and_stacked_opt_flows(self, row):
        static_frame_dir = os.path.join(self.root_dir, 'rgb', row.label, row.folder)
        opt_flow_dir_x = os.path.join(self.root_dir, 'opt_flow', 'u', row.folder)
        opt_flow_dir_y = os.path.join(self.root_dir, 'opt_flow', 'v', row.folder)
        total_frames = len(os.listdir(opt_flow_dir_x))

        # Take a random clip from that video
        end_frame = np.random.randint(self.opt_flow_len, total_frames)

        # Get the static frame
        static_frame = cv2.imread(static_frame_dir + '/frame%06d' % end_frame + '.jpg')
        static_frame = cv2.resize(static_frame, self.image_shape)
        static_frame = cv2.cvtColor(static_frame, cv2.COLOR_BGR2RGB)
        static_frame = static_frame / 255.0

        # Get the optical flow stack
        frames = range(end_frame - self.opt_flow_len + 1, end_frame + 1)  # selected optical flow frames
        opt_flow_stack = []
        # loop over frames
        for i_frame in frames:
            # horizontal components
            img = cv2.imread(opt_flow_dir_x + '/frame' + "%06d" % i_frame + '.jpg', 0)
            img = img / 255.0  # normalize pixels
            img = cv2.resize(img, self.image_shape)
            opt_flow_stack.append(img)

            # vertical components
            img2 = cv2.imread(opt_flow_dir_y + '/frame' + "%06d" % i_frame + '.jpg', 0)
            img2 = img2 / 255.0  # normalize pixels
            img2 = cv2.resize(img2, self.image_shape)
            opt_flow_stack.append(img2)

        opt_flow_stack = np.asarray(opt_flow_stack)
        opt_flow_stack = np.swapaxes(opt_flow_stack, 0, 2)

        return static_frame, opt_flow_stack
