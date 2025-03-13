import os
import argparse
import traceback
from core.train import train
from core.test import test
from core import utils
from core import dataset
from core import logger as log
import warnings
warnings.filterwarnings("ignore")

def main(config_name):
    try:
        config_path = f'configs/{config_name}.yaml'
        
        config = utils.load_config(config_path)
        experiment_dir = utils.setup_experiment(config_name)
        logger = log.Logger(experiment_dir)
        
        logger.log_message("Starting training...")
        best_acc = train(config, experiment_dir, logger)
        logger.log_message(f"Training complete. Best accuracy: {best_acc:.2f}%")
        
        logger.log_message("Starting testing...")
        save_path = test(
            config=config,
            ckpt_path=os.path.join(experiment_dir, 'best_model.pt'),
            output_dir=os.path.join(experiment_dir, 'test_results'),
            logger=logger
        )
        logger.log_message(f"Testing complete. File saved at: {save_path}")
        logger.log_message("Execution completed")

        #dataset.save_sample_images(config, experiment_dir)
        #print(f"Sample images saved to {experiment_dir}/sample_images")
    
    except Exception as e:
        logger.log_message("An error occurred during the execution:")
        logger.log_message(f"Error: {traceback.format_exc()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Test a Model')
    parser.add_argument('--config_name', type=str, default='MyResNet', help='Name of the config to use')
    args = parser.parse_args()
    
    main(args.config_name)
