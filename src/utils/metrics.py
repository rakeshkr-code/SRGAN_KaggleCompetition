import torch
import numpy as np
from typing import Tuple


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate SSIM (simplified version)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    
    sigma1_sq = torch.var(img1)
    sigma2_sq = torch.var(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim.item()


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2


def batch_psnr(sr_images: torch.Tensor, hr_images: torch.Tensor) -> float:
    """Calculate average PSNR for a batch"""
    sr_images = denormalize(sr_images)
    hr_images = denormalize(hr_images)
    
    psnr_values = []
    for sr, hr in zip(sr_images, hr_images):
        psnr = calculate_psnr(sr, hr)
        if not np.isinf(psnr):
            psnr_values.append(psnr)
    
    return np.mean(psnr_values) if psnr_values else 0.0
