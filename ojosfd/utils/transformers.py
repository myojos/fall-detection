import albumentations as alb
import numpy as np
import torch
from PIL import Image


class AlbuWrapper:  # typing: ignore
    def __init__(self, atrans: alb.BasicTransform):
        self.atrans = atrans

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.atrans(image=np.array(img))["image"]


class AlbuWrapperNumpy:  # typing: ignore
    def __init__(self, atrans: alb.BasicTransform):
        self.atrans = atrans

    def __call__(self, img):
        return self.atrans(image=img)["image"]


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


class HorizontalFlipOF(object):
    """Horizontal Flip Optical Flow in sample to Tensors."""

    def __init__(self, p=0.5, always_apply=False):
        assert isinstance(p, float)
        self.p = p
        self.always_apply = always_apply

    def __call__(self, force_apply=False, **data):
        if self.always_apply or (np.random.rand() < self.p):
            res = {}
            modified = np.copy(data["image"])
            modified[:, :, ::2] *= -1  # multiply horizontal component with -1
            modified = modified[:, ::-1, ...]
            res["image"] = np.ascontiguousarray(modified)
            return res
        return data
