#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import traceback
from datetime import datetime
from core.train import train
from core.test import test
from core.utils import setup_experiment, load_config
from core.dataset import save_sample_images
import warnings
warnings.filterwarnings("ignore")

def main():
    try:
        config_name = 'best_model'
        config_path = f'configs/{config_name}.yaml'
        
        config = load_config(config_path)
        experiment_dir, logger = setup_experiment(config)
        
        print("Starting training...")
        best_acc, metrics_df = train(config, experiment_dir, logger)
        print(f"Training complete. Best accuracy: {best_acc:.2f}%")
        
        print("Starting testing...")
        model_path = os.path.join(experiment_dir, 'best_model.pt')
        output_dir = os.path.join(experiment_dir, 'test_results')
        save_path = test(config, model_path, output_dir, logger)
        print(f"Testing complete. File saved at: {save_path}")
        
        print("Execution completed")
        #save_sample_images(config, experiment_dir)
        #print(f"Sample images saved to {experiment_dir}/sample_images")

    except Exception as e:
        print("An error occurred during the execution:")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
