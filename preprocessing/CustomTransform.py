
import numpy as np
import random

class HorizontalFlipOF(object):
    """Horizontal Flip Optical Flow in sample to Tensors."""

    def __init__(self, p=0.5, always_apply=False):
        assert isinstance(p, float)
        self.p = p
        self.always_apply = always_apply
    
    def __call__(self, force_apply=False, **data):
        if self.always_apply or (random.random() < self.p):
            res = {}
            modified = np.copy(data["image"])
            modified[:,:,::2] *= -1 # multiply horizontal component with -1
            modified = modified[:,::-1,...]
            res["image"] = np.ascontiguousarray(modified)
            return res
        return data
