from albumentations.pytorch import ToTensorV2
from albumentations import *
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations import RandomRotate90, Resize, HorizontalFlip, RandomBrightnessContrast, GaussianBlur

transform = {
        'train': Compose([
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            GaussianBlur(p=0.1),
            Resize(height=256, width=256, p=1.0),
            transforms.Normalize(),
            ToTensorV2()
        ]),
        'val': Compose([
            #HorizontalFlip(p=0.5),
            #RandomBrightnessContrast(p=0.2),
            #GaussianBlur(p=0.1),
            Resize(height=256, width=256, p=1.0),
            transforms.Normalize(),
            ToTensorV2()
        ]),
        'test': Compose([
            Resize(height=256, width=256, p=1.0),
            transforms.Normalize(),
            ToTensorV2()
        ])
}





