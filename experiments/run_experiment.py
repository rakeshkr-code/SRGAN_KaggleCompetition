import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional
import os
from tqdm import tqdm
import mlflow

from src.losses.perceptual_loss import CombinedLoss
from src.utils.metrics import batch_psnr
from src.utils.logger import Logger
from src.utils.visualization import save_comparison


class Trainer:
    """Trainer class for SRGAN"""
    
    def __init__(self, config: Dict, logger: Logger, use_mlflow: bool = True):
        self.config = config
        self.logger = logger
        self.use_mlflow = use_mlflow
        
        # Loss function
        self.criterion = CombinedLoss(
            perceptual_weight=config['loss']['perceptual_weight'],
            adversarial_weight=config['loss']['adversarial_weight'],
            pixel_weight=config['loss']['pixel_weight']
        )
        
        # Mixed precision training
        self.use_amp = config['training']['mixed_precision']
        # self.scaler = GradScaler() if self.use_amp else None
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Checkpoint management
        self.checkpoint_dir = config['checkpoint']['save_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Best model tracking
        self.best_psnr = 0.0
        self.best_epoch = 0
    
    def setup_optimizers(self, model):
        """Setup optimizers and schedulers"""
        gen_optimizer = optim.Adam(
            model.generator.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2'])
        )
        
        disc_optimizer = optim.Adam(
            model.discriminator.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2'])
        )
        
        # Learning rate schedulers
        gen_scheduler = optim.lr_scheduler.MultiStepLR(
            gen_optimizer,
            milestones=self.config['training']['lr_decay_epochs'],
            gamma=self.config['training']['lr_decay_factor']
        )
        
        disc_scheduler = optim.lr_scheduler.MultiStepLR(
            disc_optimizer,
            milestones=self.config['training']['lr_decay_epochs'],
            gamma=self.config['training']['lr_decay_factor']
        )
        
        return gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler
    
    def train_epoch(self, model, train_loader, gen_optimizer, disc_optimizer, epoch):
        """Train for one epoch"""
        model.set_mode('train')
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_psnr = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (lr_images, hr_images) in enumerate(progress_bar):
            lr_images = lr_images.to(model.device)
            hr_images = hr_images.to(model.device)
            
            # =============== Train Discriminator ===============
            disc_optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                # Generate SR images
                sr_images = model.generator(lr_images)
                
                # Discriminator predictions
                disc_pred_real = model.discriminator(hr_images)
                disc_pred_fake = model.discriminator(sr_images.detach())
                
                # Discriminator loss
                d_loss, d_loss_dict = self.criterion.discriminator_loss(
                    disc_pred_real, disc_pred_fake
                )
            
            if self.use_amp:
                self.scaler.scale(d_loss).backward()
                self.scaler.step(disc_optimizer)
            else:
                d_loss.backward()
                disc_optimizer.step()
            
            # =============== Train Generator ===============
            gen_optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                # Generate SR images again (with gradients)
                sr_images = model.generator(lr_images)
                
                # Discriminator prediction on fake images
                disc_pred_fake = model.discriminator(sr_images)
                
                # Generator loss
                g_loss, g_loss_dict = self.criterion.generator_loss(
                    sr_images, hr_images, disc_pred_fake
                )
            
            if self.use_amp:
                self.scaler.scale(g_loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(gen_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.generator.parameters(),
                    self.config['training']['gradient_clip']
                )
                
                self.scaler.step(gen_optimizer)
                self.scaler.update()
            else:
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.generator.parameters(),
                    self.config['training']['gradient_clip']
                )
                gen_optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                psnr = batch_psnr(sr_images, hr_images)
            
            total_g_loss += g_loss_dict['total_loss']
            total_d_loss += d_loss_dict['total_loss']
            total_psnr += psnr
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f"{g_loss_dict['total_loss']:.4f}",
                'D_Loss': f"{d_loss_dict['total_loss']:.4f}",
                'PSNR': f"{psnr:.2f}"
            })
        
        # Average metrics
        num_batches = len(train_loader)
        metrics = {
            'generator_loss': total_g_loss / num_batches,
            'discriminator_loss': total_d_loss / num_batches,
            'psnr': total_psnr / num_batches
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, model, val_loader, epoch):
        """Validate the model"""
        model.set_mode('eval')
        
        total_psnr = 0.0
        num_batches = 0
        
        # Save visualization samples
        save_vis = (epoch % self.config['logging']['log_images_frequency'] == 0)
        vis_saved = False
        
        for batch_idx, (lr_images, hr_images) in enumerate(tqdm(val_loader, desc="Validation")):
            lr_images = lr_images.to(model.device)
            hr_images = hr_images.to(model.device)
            
            # Generate SR images
            sr_images = model.predict(lr_images)
            
            # Calculate PSNR
            psnr = batch_psnr(sr_images, hr_images)
            total_psnr += psnr
            num_batches += 1
            
            # Save visualization
            if save_vis and not vis_saved:
                vis_path = os.path.join(
                    self.config['logging']['log_dir'],
                    f"comparison_epoch_{epoch}.png"
                )
                save_comparison(lr_images, sr_images, hr_images, vis_path)
                vis_saved = True
                
                if self.use_mlflow:
                    mlflow.log_artifact(vis_path)
        
        avg_psnr = total_psnr / num_batches
        
        return {'psnr': avg_psnr}
    
    def train(self, model, train_loader, val_loader):
        """Complete training loop"""
        self.logger.info("Starting training...")
        
        # Setup optimizers
        gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler = \
            self.setup_optimizers(model)
        
        # Training loop
        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Epoch {epoch}/{self.config['training']['epochs']}")
            self.logger.info(f"{'='*80}")
            
            # Train
            train_metrics = self.train_epoch(
                model, train_loader, gen_optimizer, disc_optimizer, epoch
            )
            
            self.logger.log_metrics(epoch, train_metrics, phase='train')
            
            # Validate
            if epoch % self.config['validation']['frequency'] == 0:
                val_metrics = self.validate(model, val_loader, epoch)
                self.logger.log_metrics(epoch, val_metrics, phase='val')
                
                # Log to MLflow
                if self.use_mlflow:
                    mlflow.log_metrics({
                        'train_g_loss': train_metrics['generator_loss'],
                        'train_d_loss': train_metrics['discriminator_loss'],
                        'train_psnr': train_metrics['psnr'],
                        'val_psnr': val_metrics['psnr']
                    }, step=epoch)
                
                # Save best model
                if val_metrics['psnr'] > self.best_psnr:
                    self.best_psnr = val_metrics['psnr']
                    self.best_epoch = epoch
                    best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                    model.save(best_path, epoch, {
                        'gen': gen_optimizer,
                        'disc': disc_optimizer
                    })
                    self.logger.info(f"✓ New best model saved (PSNR: {self.best_psnr:.4f})")
            
            # Save checkpoint
            if epoch % self.config['checkpoint']['save_frequency'] == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f'checkpoint_epoch_{epoch}.pth'
                )
                model.save(checkpoint_path, epoch, {
                    'gen': gen_optimizer,
                    'disc': disc_optimizer
                })
            
            # Update learning rates
            gen_scheduler.step()
            disc_scheduler.step()
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Training completed!")
        self.logger.info(f"Best PSNR: {self.best_psnr:.4f} (Epoch {self.best_epoch})")
        self.logger.info(f"{'='*80}")
