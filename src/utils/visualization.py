import torch
import torchvision
import matplotlib.pyplot as plt
import os
from typing import List


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2


def save_image_grid(images: List[torch.Tensor], titles: List[str], 
                   save_path: str, nrow: int = 4):
    """Save grid of images with titles"""
    fig, axes = plt.subplots(len(images), 1, figsize=(15, 5 * len(images)))
    
    if len(images) == 1:
        axes = [axes]
    
    for ax, img_batch, title in zip(axes, images, titles):
        # Denormalize and create grid
        img_batch = denormalize(img_batch)
        grid = torchvision.utils.make_grid(img_batch[:nrow], nrow=nrow, normalize=False)
        
        # Convert to numpy
        grid_np = grid.cpu().permute(1, 2, 0).numpy()
        
        ax.imshow(grid_np)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_comparison(lr_images: torch.Tensor, sr_images: torch.Tensor, 
                   hr_images: torch.Tensor, save_path: str):
    """Save comparison of LR, SR, and HR images"""
    save_image_grid(
        images=[lr_images, sr_images, hr_images],
        titles=['Low Resolution (Input)', 'Super Resolution (Generated)', 'High Resolution (Ground Truth)'],
        save_path=save_path
    )
