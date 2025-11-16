#!/usr/bin/env python3
"""
Main script for running SRGAN experiments with MLflow tracking
"""

import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
import mlflow

from src.models.srgan import SRGAN
from src.data.dataset import LowLightDataset, get_transforms
from src.utils.logger import Logger
from experiments.run_experiment import Trainer


def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_mlflow(config: dict, run_name: str = None):
    """Setup MLflow tracking"""
    if config['logging']['mlflow_tracking']:
        mlflow.set_tracking_uri(config['logging']['mlflow_uri'])
        mlflow.set_experiment(config['logging']['experiment_name'])
        
        if run_name:
            mlflow.start_run(run_name=run_name)
        else:
            mlflow.start_run()
        
        # Log config parameters
        mlflow.log_params({
            'batch_size': config['data']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'epochs': config['training']['epochs'],
            'num_residual_blocks': config['model']['generator']['num_residual_blocks'],
            'base_channels': config['model']['generator']['base_channels'],
            'perceptual_weight': config['loss']['perceptual_weight'],
            'adversarial_weight': config['loss']['adversarial_weight'],
            'pixel_weight': config['loss']['pixel_weight']
        })
        
        return True
    return False


def prepare_dataloaders(config: dict):
    """Prepare train and validation dataloaders"""
    
    # Get transforms
    transform_lr_train, transform_hr_train = get_transforms('train')
    transform_lr_val, transform_hr_val = get_transforms('val')
    
    # Create datasets
    train_dataset = LowLightDataset(
        noisy_dir=config['data']['train_noisy_dir'],
        gt_dir=config['data']['train_gt_dir'],
        transform_lr=transform_lr_train,
        transform_hr=transform_hr_train
    )
    
    val_dataset = LowLightDataset(
        noisy_dir=config['data']['val_noisy_dir'],
        gt_dir=config['data']['val_gt_dir'],
        transform_lr=transform_lr_val,
        transform_hr=transform_hr_val
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return train_loader, val_loader


def main(args):
    """Main training function"""
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Setup logger
    logger = Logger(
        log_dir=config['logging']['log_dir'],
        experiment_name=config['logging']['experiment_name']
    )
    logger.log_config(config)
    
    # Setup MLflow
    use_mlflow = setup_mlflow(config, run_name=args.run_name) if not args.no_mlflow else False
    
    # Prepare dataloaders
    logger.info("Preparing datasets...")
    train_loader, val_loader = prepare_dataloaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Initialize model
    logger.info("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() and config['device']['use_cuda'] else "cpu")
    model = SRGAN(config).to_device(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1 and len(config['device']['device_ids']) > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model.generator = torch.nn.DataParallel(
            model.generator, 
            device_ids=config['device']['device_ids']
        )
        model.discriminator = torch.nn.DataParallel(
            model.discriminator,
            device_ids=config['device']['device_ids']
        )
    
    # Initialize trainer
    trainer = Trainer(config, logger, use_mlflow=use_mlflow)
    
    # Train model
    model.fit(train_loader, val_loader, trainer)
    
    # End MLflow run
    if use_mlflow:
        mlflow.end_run()
    
    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRGAN Training Script")
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--run-name', type=str, default=None,
                       help='MLflow run name')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow logging')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    
    args = parser.parse_args()
    main(args)
