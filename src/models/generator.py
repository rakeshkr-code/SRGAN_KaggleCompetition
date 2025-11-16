import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block with skip connections"""
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class UpsampleBlock(nn.Module):
    """Upsampling block using sub-pixel convolution"""
    def __init__(self, in_channels: int, upscale_factor: int = 2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), 
                             kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.pixel_shuffle(self.conv(x)))


class Generator(nn.Module):
    """SRGAN Generator for 4x Super-Resolution + Denoising"""
    def __init__(self, in_channels: int = 3, base_channels: int = 64, 
                 num_residual_blocks: int = 16, upscale_factor: int = 4):
        super(Generator, self).__init__()
        
        # Initial feature extraction
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=9, 
                              stride=1, padding=4)
        self.prelu = nn.PReLU()
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_residual_blocks)]
        )
        
        # Post-residual convolution
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, 
                              stride=1, padding=1)
        self.bn = nn.BatchNorm2d(base_channels)
        
        # Upsampling blocks (for 4x: 2x -> 2x)
        num_upsample_blocks = int(torch.log2(torch.tensor(upscale_factor)).item())
        self.upsample_blocks = nn.Sequential(
            *[UpsampleBlock(base_channels, upscale_factor=2) 
              for _ in range(num_upsample_blocks)]
        )
        
        # Final output layer
        self.conv3 = nn.Conv2d(base_channels, in_channels, kernel_size=9, 
                              stride=1, padding=4)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial feature extraction
        out1 = self.prelu(self.conv1(x))
        
        # Residual learning
        out2 = self.res_blocks(out1)
        out2 = self.bn(self.conv2(out2))
        
        # Global skip connection
        out3 = out2 + out1
        
        # Upsampling
        out4 = self.upsample_blocks(out3)
        
        # Final output
        out = self.tanh(self.conv3(out4))
        return out
