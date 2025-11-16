import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block for discriminator"""
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, use_bn: bool = True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.bn(self.conv(x)))


class Discriminator(nn.Module):
    """SRGAN Discriminator with adaptive pooling"""
    def __init__(self, in_channels: int = 3, base_channels: int = 64, 
                 num_blocks: int = 8):
        super(Discriminator, self).__init__()
        
        # Initial convolution (no BN)
        layers = [ConvBlock(in_channels, base_channels, stride=1, use_bn=False)]
        
        # Build convolutional blocks
        channels = base_channels
        for i in range(1, num_blocks):
            stride = 2 if i % 2 == 0 else 1
            out_channels = min(channels * 2, 512) if stride == 2 else channels
            layers.append(ConvBlock(channels, out_channels, stride=stride, use_bn=True))
            channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.adaptive_pool(out)
        out = self.classifier(out)
        return out
