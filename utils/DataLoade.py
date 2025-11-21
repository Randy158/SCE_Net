from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from pathlib import Path

class MedSegment(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, image_suffix='.jpg', label_suffix='.png'):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix

        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(self.image_suffix)]
        
        self.label_files = {Path(f).stem: f for f in os.listdir(label_dir) if f.endswith(self.label_suffix)}

        self.img_name = [
            img for img in self.img_files if Path(img).stem in self.label_files
        ]
        
    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        img_name = self.img_name[index]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, self.label_files[Path(img_name).stem])

       
        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path)).astype(np.float32)  # 标签转换为 float32 或 np.uint8

        if self.transform:
            res = self.transform(image=image, mask=label)
            image, label = res['image'], res['mask']

        return image, label
