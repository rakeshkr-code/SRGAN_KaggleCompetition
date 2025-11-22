# #!/usr/bin/env python3
# """Inference script for generating predictions"""

# import os
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from PIL import Image
# import yaml
# import argparse

# from src.models.srgan import SRGAN
# from src.data.dataset import LowLightDataset, get_transforms
# from src.utils.metrics import denormalize


# def generate_predictions(model, test_loader, output_dir, device):
#     """Generate predictions on test set"""
#     model.set_mode('eval')
#     os.makedirs(output_dir, exist_ok=True)
    
#     with torch.no_grad():
#         for lr_images, filenames in tqdm(test_loader, desc="Generating predictions"):
#             lr_images = lr_images.to(device)
            
#             # Generate SR images
#             sr_images = model.predict(lr_images)
#             sr_images = denormalize(sr_images).cpu()
            
#             # Save images
#             for sr_img, filename in zip(sr_images, filenames):
#                 # Convert to PIL Image
#                 sr_img_np = (sr_img.permute(1, 2, 0).numpy() * 255).astype('uint8')
#                 pil_img = Image.fromarray(sr_img_np)
                
#                 # Save
#                 output_path = os.path.join(output_dir, filename)
#                 pil_img.save(output_path)
    
#     print(f"✓ Predictions saved to: {output_dir}")


# def main(args):
#     # Load config
#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)
    
#     # Prepare test dataloader
#     transform_lr, _ = get_transforms('test')
#     test_dataset = LowLightDataset(
#         noisy_dir=config['data']['test_noisy_dir'],
#         transform_lr=transform_lr,
#         return_filename=True
#     )
    
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=4
#     )
    
#     # Load model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SRGAN(config).to_device(device)
#     model.load(args.checkpoint)
    
#     # Generate predictions
#     generate_predictions(model, test_loader, args.output_dir, device)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="SRGAN Inference Script")
    
#     parser.add_argument('--config', type=str, required=True,
#                        help='Path to config file')
#     parser.add_argument('--checkpoint', type=str, required=True,
#                        help='Path to model checkpoint')
#     parser.add_argument('--output-dir', type=str, default='outputs/predictions',
#                        help='Output directory for predictions')
#     parser.add_argument('--batch-size', type=int, default=16,
#                        help='Batch size for inference')
    
#     args = parser.parse_args()
#     main(args)


# -----------------------------------------------------------------------------------------

#!/usr/bin/env python3
"""
Inference script for SRGAN - Generate SR images from test set
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

from src.models.srgan import SRGAN
from src.data.dataset import LowLightDataset, get_transforms


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image"""
    # Denormalize
    tensor = denormalize(tensor)
    
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy [H, W, C]
    img_np = tensor.cpu().permute(1, 2, 0).numpy()
    
    # Convert to uint8 [0, 255]
    img_np = (img_np * 255).astype(np.uint8)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img_np)
    
    return pil_img


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Load trained SRGAN model"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = SRGAN(config).to_device(device)
    
    # Load checkpoint
    model.load(checkpoint_path)
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    
    return model, config


def generate_predictions(model, test_loader, output_dir, device, save_lr=False):
    """Generate SR predictions and save images"""
    
    model.set_mode('eval')
    os.makedirs(output_dir, exist_ok=True)
    
    if save_lr:
        lr_output_dir = os.path.join(output_dir, 'lr_inputs')
        os.makedirs(lr_output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Generating Super-Resolution Images")
    print(f"{'='*80}")
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Generating SR images"):
            
            # Handle both (lr_images, filenames) and (lr_images, hr_images, filenames)
            if len(batch_data) == 2:
                lr_images, filenames = batch_data
            else:
                lr_images, _, filenames = batch_data
            
            lr_images = lr_images.to(device)
            
            # Generate SR images
            sr_images = model.predict(lr_images)
            
            # Save each image in the batch
            for sr_tensor, lr_tensor, filename in zip(sr_images, lr_images, filenames):
                # Convert SR tensor to PIL Image
                sr_pil = tensor_to_image(sr_tensor)
                
                # Save SR image
                output_path = os.path.join(output_dir, filename)
                sr_pil.save(output_path, quality=95)
                
                # Optionally save LR input for comparison
                if save_lr:
                    lr_pil = tensor_to_image(lr_tensor)
                    lr_output_path = os.path.join(lr_output_dir, filename)
                    lr_pil.save(lr_output_path, quality=95)
    
    print(f"\n✓ All predictions saved to: {output_dir}")
    print(f"✓ Total images generated: {len(test_loader.dataset)}")


# def generate_single_image(model, image_path, output_path, device):
#     """Generate SR image from a single input image"""
    
#     model.set_mode('eval')
    
#     # Load and preprocess image
#     transform_lr, _ = get_transforms('test')
#     lr_image = Image.open(image_path).convert('RGB')
#     lr_tensor = transform_lr(lr_image).unsqueeze(0).to(device)  # Add batch dimension
    
#     print(f"\nProcessing: {image_path}")
#     print(f"Input shape: {lr_tensor.shape}")
    
#     # Generate SR image
#     with torch.no_grad():
#         sr_tensor = model.predict(lr_tensor)
    
#     print(f"Output shape: {sr_tensor.shape}")
    
#     # Convert to PIL and save
#     sr_pil = tensor_to_image(sr_tensor.squeeze(0))  # Remove batch dimension
#     sr_pil.save(output_path, quality=95)
    
#     print(f"✓ SR image saved to: {output_path}")
    
#     # Also save input for comparison
#     input_comparison_path = output_path.replace('.', '_input.')
#     lr_pil = tensor_to_image(lr_tensor.squeeze(0))
#     lr_pil.save(input_comparison_path, quality=95)
#     print(f"✓ Input image saved to: {input_comparison_path}")

def generate_single_image(model, image_path, output_path, device):
    """Generate SR image from a single input image"""
    
    model.set_mode('eval')
    
    # Validate input path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    # Load and preprocess image
    transform_lr, _ = get_transforms('test')
    
    try:
        lr_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")
    
    lr_tensor = transform_lr(lr_image).unsqueeze(0).to(device)  # Add batch dimension
    
    print(f"\nProcessing: {image_path}")
    print(f"Input shape: {lr_tensor.shape}")
    
    # Generate SR image
    with torch.no_grad():
        sr_tensor = model.predict(lr_tensor)
    
    print(f"Output shape: {sr_tensor.shape}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")
    
    # Convert to PIL and save SR image
    sr_pil = tensor_to_image(sr_tensor.squeeze(0))  # Remove batch dimension
    sr_pil.save(output_path, quality=95)
    print(f"✓ SR image saved to: {output_path}")
    
    # Save input for comparison (properly handle path)
    base_path, ext = os.path.splitext(output_path)
    input_comparison_path = f"{base_path}_input{ext}"
    
    lr_pil = tensor_to_image(lr_tensor.squeeze(0))
    lr_pil.save(input_comparison_path, quality=95)
    print(f"✓ Input image saved to: {input_comparison_path}")
    
    # Print upscaling factor
    scale_h = sr_pil.size[1] / lr_image.size[1]
    scale_w = sr_pil.size[0] / lr_image.size[0]
    print(f"✓ Upscaling: {lr_image.size} → {sr_pil.size} ({scale_w:.1f}× width, {scale_h:.1f}× height)")



def main(args):
    """Main inference function"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.config, args.checkpoint, device)
    
    # Single image mode
    if args.input_image:
        generate_single_image(model, args.input_image, args.output, device)
        return
    
    # Batch inference mode (test set)
    print(f"\nLoading test dataset from: {args.test_dir or config['data']['test_noisy_dir']}")
    
    # Get transforms
    transform_lr, transform_hr = get_transforms('test')
    
    # Create test dataset
    test_dataset = LowLightDataset(
        noisy_dir=args.test_dir or config['data']['test_noisy_dir'],
        gt_dir=None,  # No ground truth for test set
        transform_lr=transform_lr,
        return_filename=True
    )
    
    print(f"Found {len(test_dataset)} test images")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Generate predictions
    generate_predictions(model, test_loader, args.output_dir, device, args.save_lr)
    
    print(f"\n{'='*80}")
    print("Inference completed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRGAN Inference Script")
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    
    # Mode selection
    parser.add_argument('--input-image', type=str, default=None,
                       help='Path to single input image (for single image mode)')
    parser.add_argument('--output', type=str, default='output_sr.png',
                       help='Output path for single image mode')
    
    # Batch inference arguments
    parser.add_argument('--test-dir', type=str, default=None,
                       help='Test dataset directory (overrides config)')
    parser.add_argument('--output-dir', type=str, default='./outputs/predictions',
                       help='Output directory for batch predictions')
    
    # Optional arguments
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save-lr', action='store_true',
                       help='Save LR input images for comparison')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU inference (disable CUDA)')
    
    args = parser.parse_args()
    main(args)
