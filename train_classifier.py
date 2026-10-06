import hydra
import wandb
import os
import torch
import time
from omegaconf import OmegaConf
import torch.nn as nn

from data_processing.data_loaders import get_data_loaders
from classifiers.get_model import get_model
from defenses.get_defense import get_defense

from utils import wandb_helpers
from evaluate_classifier import calculate_top5_accuracy

@hydra.main(config_path="configuration/classifier", config_name="config.yaml", version_base="1.3")
def train_classifier(config):
    
    if torch.cuda.is_available() and config.training.device != "cuda":
        question = f"\nCuda is available but not configured from command to be used! Do you wish to use cuda instead of {config.training.device}?\nType y to use cuda, enter if not:"
        
        use_cuda = input(question)
        
        if use_cuda.lower().strip() == "y":
            config.training.device = 'cuda'
            
    if config.training.wandb.track:
        wandb_helpers.wandb_init(config)
    
    # Load the data
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # Load the model
    model = get_model(config)
    
    # Load defense
    model = get_defense(config=config, model=model)
    
    # Load trained model weights if given
    if config.load_from_wandb_id:
        
        model_weights_path = wandb_helpers.get_weights(
                                                        entity   = config.training.wandb.entity,
                                                        project  = config.training.wandb.project,
                                                        run_id = config.load_from_wandb_id,
                                                        load_best_model=config.training.load_best_model
        )
        
        model.load_model(model_weights_path)

    # Train the model
    model.train_model(train_loader, val_loader)

    test_loss, test_accuracy = model.evaluate(test_loader)
    top5_accuracy = calculate_top5_accuracy(model, test_loader, device=config.training.device)
    
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Top-5 Accuracy: {top5_accuracy}")
    
    if config.training.wandb.track:
        wandb.log({"test_loss": test_loss})
        wandb.log({"test_accuracy": test_accuracy})
        wandb.log({"top5_accuracy": top5_accuracy})
        
    # Save the configuration for later use   
    OmegaConf.save(config, f"classifiers/saved_configs/{model.save_as}.yaml")

    # Save weights in wandb
    if config.training.wandb.track:
        # Save the best model
        model_path = f"classifiers/saved_models/{model.save_as}"
        wandb.save(model_path)
        
        # Save the last epoch model
        base, ext = os.path.splitext(model_path)
        last_epoch_path = f"{base}_last_epoch{ext}"
        wandb.save(last_epoch_path)

        if config.training.save_many_accs:
            for thresh in config.training.early_stopping_accuracy_thresholds:
                thresh_model_path = f"{base}_acc{thresh}{ext}"
                if os.path.exists(thresh_model_path):
                    wandb.save(thresh_model_path)
                    print(f"wandb: Saved model for threshold {thresh} at {thresh_model_path}")
                else:
                    print(f"File does not exist for threshold {thresh}: {thresh_model_path}")

    
if __name__ == "__main__":
    train_classifier()