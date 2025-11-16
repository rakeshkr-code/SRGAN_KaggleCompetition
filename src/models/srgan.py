import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .generator import Generator
from .discriminator import Discriminator


class SRGAN(nn.Module):
    """Complete SRGAN model with training/inference methods"""
    
    def __init__(self, config: Dict):
        super(SRGAN, self).__init__()
        
        # Initialize generator and discriminator
        self.generator = Generator(
            in_channels=config['model']['generator']['in_channels'],
            base_channels=config['model']['generator']['base_channels'],
            num_residual_blocks=config['model']['generator']['num_residual_blocks'],
            upscale_factor=config['model']['generator']['upscale_factor']
        )
        
        self.discriminator = Discriminator(
            in_channels=config['model']['discriminator']['in_channels'],
            base_channels=config['model']['discriminator']['base_channels'],
            num_blocks=config['model']['discriminator']['num_blocks']
        )
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def to_device(self, device: torch.device):
        """Move model to device"""
        self.device = device
        self.generator.to(device)
        self.discriminator.to(device)
        return self
    
    def set_mode(self, mode: str):
        """Set training/evaluation mode"""
        if mode == 'train':
            self.generator.train()
            self.discriminator.train()
        elif mode == 'eval':
            self.generator.eval()
            self.discriminator.eval()
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def predict(self, lr_images: torch.Tensor) -> torch.Tensor:
        """Generate SR images from LR inputs"""
        self.generator.eval()
        with torch.no_grad():
            sr_images = self.generator(lr_images)
        return sr_images
    
    def fit(self, train_loader, val_loader, trainer):
        """Training wrapper - delegates to Trainer class"""
        return trainer.train(self, train_loader, val_loader)
    
    def save(self, path: str, epoch: int, optimizers: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'config': self.config
        }
        
        if optimizers:
            checkpoint['gen_optimizer_state_dict'] = optimizers['gen'].state_dict()
            checkpoint['disc_optimizer_state_dict'] = optimizers['disc'].state_dict()
        
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")
    
    def load(self, path: str, load_optimizers: bool = False) -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        print(f"✓ Checkpoint loaded: {path} (Epoch {checkpoint['epoch']})")
        
        if load_optimizers:
            return {
                'epoch': checkpoint['epoch'],
                'gen_optimizer': checkpoint.get('gen_optimizer_state_dict'),
                'disc_optimizer': checkpoint.get('disc_optimizer_state_dict')
            }
        
        return {'epoch': checkpoint['epoch']}
