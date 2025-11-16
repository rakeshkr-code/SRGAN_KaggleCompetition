#!/usr/bin/env python3
"""Inference script for generating predictions"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import yaml
import argparse

from src.models.srgan import SRGAN
from src.data.dataset import LowLightDataset, get_transforms
from src.utils.metrics import denormalize


def generate_predictions(model, test_loader, output_dir, device):
    """Generate predictions on test set"""
    model.set_mode('eval')
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for lr_images, filenames in tqdm(test_loader, desc="Generating predictions"):
            lr_images = lr_images.to(device)
            
            # Generate SR images
            sr_images = model.predict(lr_images)
            sr_images = denormalize(sr_images).cpu()
            
            # Save images
            for sr_img, filename in zip(sr_images, filenames):
                # Convert to PIL Image
                sr_img_np = (sr_img.permute(1, 2, 0).numpy() * 255).astype('uint8')
                pil_img = Image.fromarray(sr_img_np)
                
                # Save
                output_path = os.path.join(output_dir, filename)
                pil_img.save(output_path)
    
    print(f"✓ Predictions saved to: {output_dir}")


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Prepare test dataloader
    transform_lr, _ = get_transforms('test')
    test_dataset = LowLightDataset(
        noisy_dir=config['data']['test_noisy_dir'],
        transform_lr=transform_lr,
        return_filename=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRGAN(config).to_device(device)
    model.load(args.checkpoint)
    
    # Generate predictions
    generate_predictions(model, test_loader, args.output_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRGAN Inference Script")
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/predictions',
                       help='Output directory for predictions')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    main(args)
