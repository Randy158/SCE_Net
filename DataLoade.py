from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class MedSegment(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, image_suffix: str = '.jpg',label_suffix: str = '.png'):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.img_name = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        image = np.array(Image.open(os.path.join(self.img_dir, self.img_name[index])).convert("RGB"))
        label = np.array(Image.open(os.path.join(self.label_dir, self.img_name[index].replace(self.image_suffix, self.label_suffix)))).astype(np.float16)

        if self.transform is not None:
            res = self.transform(image=image, mask=label)

        return res


