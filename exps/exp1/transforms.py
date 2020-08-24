
import torch
from torchvision import transforms, utils
import albumentations as alb
import albumentations.augmentations.transforms as aat

class AlbuWrapperNumpy:  # typing: ignore
    def __init__(self, atrans: alb.BasicTransform):
        self.atrans = atrans

    def __call__(self, img):
        return self.atrans(image=img)["image"]

alb_transforms = alb.Compose(
        [
            alb.Resize(256, 256, always_apply=True),
            alb.RandomCrop(244, 244, always_apply=True),
            aat.HorizontalFlip(),
            aat.Cutout(2, 10, 10)
        ])
alb_rescale = alb.Resize(244, 244, always_apply=True)

transform = transforms.Compose(
  [transforms.ToTensor(),
    transforms.Normalize(0.449, 0.226)])

test_transforms = transforms.Compose(
  [AlbuWrapperNumpy(alb_rescale), transform])
train_transforms = transforms.Compose(
  [AlbuWrapperNumpy(alb_transforms), transform])
