import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from typing import Tuple, Dict


class PerceptualLoss(nn.Module):
    """VGG19-based perceptual loss"""
    
    def __init__(self, layer_idx: int = 36, normalize_input: bool = True):
        super(PerceptualLoss, self).__init__()
        
        # Load pretrained VGG19
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features)[:layer_idx]).eval()
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.normalize_input = normalize_input
        
        # VGG normalization parameters (registered as buffers for device compatibility)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize images for VGG"""
        # Denormalize from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Normalize with ImageNet stats (mean and std are already on same device as x)
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            sr = self.normalize(sr)
            hr = self.normalize(hr)
        
        sr_features = self.features(sr)
        hr_features = self.features(hr)
        
        loss = nn.functional.mse_loss(sr_features, hr_features)
        return loss


class CombinedLoss(nn.Module):
    """Combined loss for SRGAN training"""
    
    def __init__(self, perceptual_weight: float = 1.0, 
                 adversarial_weight: float = 0.001,
                 pixel_weight: float = 1.0,
                 vgg_layer: int = 36):
        super(CombinedLoss, self).__init__()
        
        self.perceptual_loss = PerceptualLoss(layer_idx=vgg_layer)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.pixel_loss = nn.MSELoss()
        
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.pixel_weight = pixel_weight
    
    def to(self, device):
        """Override to method to ensure all components are moved to device"""
        super().to(device)
        self.perceptual_loss.to(device)
        return self
    
    def generator_loss(self, sr_images: torch.Tensor, hr_images: torch.Tensor,
                      disc_pred_fake: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute generator loss"""
        
        # Pixel-wise loss (for PSNR)
        pixel_loss = self.pixel_loss(sr_images, hr_images)
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(sr_images, hr_images)
        
        # Adversarial loss
        batch_size = disc_pred_fake.size(0)
        real_labels = torch.ones(batch_size, 1, device=disc_pred_fake.device)
        adversarial_loss = self.adversarial_loss(disc_pred_fake, real_labels)
        
        # Total generator loss
        total_loss = (self.pixel_weight * pixel_loss + 
                     self.perceptual_weight * perceptual_loss +
                     self.adversarial_weight * adversarial_loss)
        
        loss_dict = {
            'pixel_loss': pixel_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'adversarial_loss': adversarial_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def discriminator_loss(self, disc_pred_real: torch.Tensor, 
                          disc_pred_fake: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute discriminator loss"""
        
        batch_size = disc_pred_real.size(0)
        real_labels = torch.ones(batch_size, 1, device=disc_pred_real.device)
        fake_labels = torch.zeros(batch_size, 1, device=disc_pred_fake.device)
        
        real_loss = self.adversarial_loss(disc_pred_real, real_labels)
        fake_loss = self.adversarial_loss(disc_pred_fake, fake_labels)
        
        total_loss = (real_loss + fake_loss) / 2
        
        loss_dict = {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict

