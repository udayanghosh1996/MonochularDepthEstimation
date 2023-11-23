import torch
import torch.nn as nn
import torchvision.models as models


class MonoDepthModel(nn.Module):
    def __init__(self):
        super(MonoDepthModel, self).__init__()

        resnet18 = models.resnet18(pretrained=True)
        self.enc1 = nn.Sequential(resnet18.conv1, resnet18.bn1, resnet18.relu, resnet18.maxpool, resnet18.layer1)
        self.enc2 = resnet18.layer2
        self.enc3 = resnet18.layer3
        self.enc4 = resnet18.layer4

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        dec4 = self.upconv4(x4)
        dec3 = self.upconv3(dec4)
        dec2 = self.upconv2(dec3)
        dec1 = self.upconv1(dec2)

        depth_map = self.output_layer(dec1)

        return depth_map
