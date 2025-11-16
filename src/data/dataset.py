import os
from typing import Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LowLightDataset(Dataset):
    """Dataset for low-light image super-resolution"""
    
    def __init__(self, noisy_dir: str, gt_dir: Optional[str] = None, 
                 transform_lr=None, transform_hr=None, return_filename: bool = False):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.return_filename = return_filename
        
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if gt_dir:
            self.gt_files = sorted([f for f in os.listdir(gt_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
            assert len(self.noisy_files) == len(self.gt_files), \
                "Mismatch between noisy and GT image counts"
        else:
            self.gt_files = None
        
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self) -> int:
        return len(self.noisy_files)

    def __getitem__(self, idx: int) -> Tuple:
        # Load noisy (LR) image
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        noisy_image = Image.open(noisy_path).convert('RGB')
        
        if self.transform_lr:
            noisy_image = self.transform_lr(noisy_image)
        
        # Load ground truth (HR) image if available
        if self.gt_dir:
            gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
            gt_image = Image.open(gt_path).convert('RGB')
            
            if self.transform_hr:
                gt_image = self.transform_hr(gt_image)
            
            if self.return_filename:
                return noisy_image, gt_image, self.noisy_files[idx]
            return noisy_image, gt_image
        
        if self.return_filename:
            return noisy_image, self.noisy_files[idx]
        return noisy_image


def get_transforms(mode: str = 'train'):
    """Get image transforms for LR and HR images"""
    
    if mode == 'train':
        transform_lr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        transform_hr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    else:  # val/test
        transform_lr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        transform_hr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    return transform_lr, transform_hr
