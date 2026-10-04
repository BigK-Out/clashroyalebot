"""Per-frame ResNet-18 features, pooled over time (mean + attention), one linear head."""
import torch
import torch.nn as nn
import torchvision

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class PlayNet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        r.fc = nn.Identity()
        self.backbone = r
        self.attn = nn.Linear(512, 1)
        self.head = nn.Linear(1024, n_classes)
        self.register_buffer("mean", MEAN)
        self.register_buffer("std", STD)

    def forward(self, clip):  # [B, T, 3, H, W], 0..1
        b, t = clip.shape[:2]
        x = ((clip - self.mean) / self.std).flatten(0, 1)
        f = self.backbone(x).view(b, t, 512)
        w = torch.softmax(self.attn(f), dim=1)
        return self.head(torch.cat([f.mean(1), (w * f).sum(1)], 1))
