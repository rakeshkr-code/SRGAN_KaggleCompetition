#!/usr/bin/env python3
"""
Hyperparameter sweep script with MLflow
"""

import argparse
import yaml
import copy
import itertools
from main import main as train_main


class SimpleArgs:
    """Simple argument class for passing to main()"""
    def __init__(self, config_path, run_name, batch_size, lr, epochs, no_mlflow=False):
        self.config = config_path
        self.run_name = run_name
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.no_mlflow = no_mlflow


def run_sweep(config_path: str, sweep_params: dict):
    """Run hyperparameter sweep"""
    
    # Generate all combinations
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"Running {len(combinations)} experiments...")
    
    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))
        
        run_name = f"sweep_{idx}_{'.'.join([f'{k}={v}' for k, v in params.items()])}"
        print(f"\n{'='*80}")
        print(f"Experiment {idx}/{len(combinations)}: {run_name}")
        print(f"{'='*80}")
        
        # Create args
        args = SimpleArgs(
            config_path=config_path,
            run_name=run_name,
            batch_size=params.get('batch_size'),
            lr=params.get('learning_rate'),
            epochs=params.get('epochs')
        )
        
        # Run experiment
        try:
            train_main(args)
        except Exception as e:
            print(f"Error in experiment {idx}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("Sweep completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Sweep Script")
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--sweep-config', type=str, required=True,
                       help='Path to sweep configuration YAML')
    
    args = parser.parse_args()
    
    # Load sweep configuration
    with open(args.sweep_config, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    run_sweep(args.config, sweep_config['parameters'])
