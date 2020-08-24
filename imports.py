# system
from __future__ import print_function, division
import os
import time
import random
from tqdm import tqdm

# common
import pandas as pd
import yaml
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

# opencv
import cv2

# torch
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# augment
import albumentations as alb
import albumentations.augmentations.transforms as aat
import numpy as np
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
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
