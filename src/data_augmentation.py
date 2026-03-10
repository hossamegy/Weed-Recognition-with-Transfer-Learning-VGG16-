
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DataAugmentation:
    def train_transform(self, MEAN, STD):
        return transforms.Compose([
            transforms.Resize((256, 256)),         
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45), 
            transforms.ColorJitter(             
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        
    def eval_transform(self, MEAN, STD):
        return transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label