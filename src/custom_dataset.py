import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self, data_dir_path, transform=None, target_transform=None,
                 min_resolution=300):
        self.data_dir_path = data_dir_path
        self.transform = transform
        self.target_transform = target_transform

        self.classes = sorted(os.listdir(data_dir_path))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(data_dir_path, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            for img_path in glob.glob(os.path.join(cls_folder, "*")):
                if not img_path.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    continue
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                    if w >= min_resolution and h >= min_resolution:
                        self.samples.append((img_path,
                                             self.class_to_idx[cls_name]))
                except Exception:
                    pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
