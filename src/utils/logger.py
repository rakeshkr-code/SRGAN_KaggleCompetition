import os
import logging
from datetime import datetime
from typing import Dict, Any


class Logger:
    """Custom logger for experiments"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(experiment_name)
        self.logger.info(f"Logger initialized: {log_file}")
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = 'train'):
        """Log metrics for an epoch"""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"[{phase.upper()}] Epoch {epoch} | {metrics_str}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration"""
        self.logger.info("=" * 80)
        self.logger.info("CONFIGURATION")
        self.logger.info("=" * 80)
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 80)
