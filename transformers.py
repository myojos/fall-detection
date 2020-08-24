import numpy as np
import torch


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        flow, rgb, y = sample['flow'], sample['rgb'], sample['y']
        # swap axis because
        # numpy image: N x H x W x C
        # torch image: N x C X H X W
        return {'flow': torch.from_numpy(np.asarray(flow).transpose((0, 3, 1, 2))),
                'rgb': torch.from_numpy(np.asarray(rgb).transpose((0, 3, 1, 2))),
                'y': torch.from_numpy(np.asarray(y))}


class RandomHorizontalFlip:
    """Randomly flips np arrays horizontaly"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        flow, rgb, y = sample['flow'], sample['rgb'], sample['y']
        # torch images: N x C x H x W
        if np.random.rand() < self.p:
            flow = torch.flip(flow, dims=(-1,))
            rgb = torch.flip(rgb, dims=(-1,))

        return {'flow': flow, 'rgb': rgb, 'y': y}
