import torch.nn as nn
from torchvision import models

class WeedVGG16(nn.Module):
    def __init__(self, num_classes=12, dropout=0.2):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        self.features = vgg.features[:24] 
        for param in self.features.parameters():
            param.requires_grad = False

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)    
        x = self.global_avg_pool(x) 
        x = self.classifier(x) 
        return x
