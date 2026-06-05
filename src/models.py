import torch
import torch.nn as nn
from src.config import NUM_CLASSES, DEVICE

BACKBONES = {
    "mobilenet_v3_small": ("mobilenet_v3_small", "classifier", 0.2),
    "resnet50": ("resnet50", "fc", 0.5),
    "efficientnet_b3": ("efficientnet_b3", "classifier", 0.5),
}

def get_classifier(pretrained=True, backbone="mobilenet_v3_small"):
    import torchvision.models as models
    name, head_attr, dropout = BACKBONES[backbone]
    weights = "IMAGENET1K_V1" if pretrained else None
    model = getattr(models, name)(weights=weights)

    head = getattr(model, head_attr)
    if isinstance(head, nn.Sequential):
        for layer in reversed(head):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                break
    else:
        in_features = head.in_features
    setattr(model, head_attr, nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, NUM_CLASSES),
    ))
    return model.to(DEVICE)

class UNetMini(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        def conv_block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )
        def up_block(c_in, c_out):
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, 2, stride=2),
            )

        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bridge = conv_block(128, 256)
        self.up1 = up_block(256, 128)
        self.dec1 = conv_block(256, 128)
        self.up2 = up_block(128, 64)
        self.dec2 = conv_block(128, 64)
        self.up3 = up_block(64, 32)
        self.dec3 = conv_block(64, 32)
        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bridge(self.pool(e3))
        d1 = self.dec1(torch.cat([self.up1(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e1], dim=1))
        return self.out(d3)

def get_segmenter(backbone="resnet34", img_size=384):
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES + 1,
            activation=None,
        )
    except ImportError:
        print("  segmentation_models_pytorch not found, falling back to UNetMini")
        model = UNetMini(in_channels=3, out_channels=NUM_CLASSES + 1)
    return model.to(DEVICE)
