"""Plain 2D U-Net: five levels, 64 initial channels, no pretrained backbone.

Uses padded convolutions to preserve the input grid instead of the original
paper's valid-convolution cropping. No attention or normalization layers.
"""
import torch
from torch import nn
from torch.nn import functional as F


def double_conv(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
    )


class PlainUNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [64, 128, 256, 512, 1024]
        self.encoders = nn.ModuleList([
            double_conv(3 if index == 0 else channels[index - 1], width)
            for index, width in enumerate(channels)
        ])
        self.pool = nn.MaxPool2d(2)
        reversed_channels = list(reversed(channels))
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(source, destination, 2, stride=2)
            for source, destination in zip(reversed_channels[:-1], reversed_channels[1:])
        ])
        self.decoders = nn.ModuleList([
            double_conv(width * 2, width) for width in reversed_channels[1:]
        ])
        self.output = nn.Conv2d(64, 1, 1)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, image):
        skips = []
        value = image
        for index, encoder in enumerate(self.encoders):
            if index:
                value = self.pool(value)
            value = encoder(value)
            skips.append(value)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips[:-1])):
            value = upconv(value)
            if value.shape[-2:] != skip.shape[-2:]:
                value = F.interpolate(value, skip.shape[-2:], mode="bilinear", align_corners=False)
            value = decoder(torch.cat([skip, value], dim=1))
        return self.output(value)
