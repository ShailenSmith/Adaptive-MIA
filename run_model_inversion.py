from omegaconf import OmegaConf
import omegaconf
import hydra
import wandb
import os
import torch
import sys
import time

from classifiers.get_model import get_model
from defenses.get_defense import get_defense
from data_processing.data_loaders import get_data_loaders
from data_processing.datasets import get_datasets
from data_processing.data_augmentation import get_transforms

import Plug_and_Play_Attacks.utils.attack_config_parser as ppa_config_parser

import model_inversion.plug_and_play.modify_to_ppa_repo as ppa_modify # import model_compatibility_wrapper, convert_configs
import model_inversion.plug_and_play.attack as ppa

import IF_GMI.utils.attack_config_parser as if_gmi_config_parser
import model_inversion.if_gmi.attack as if_gmi
import model_inversion.if_gmi.modify_to_repo as if_gmi_modify

import PPDG_MI.high_resolution.utils.attack_config_parser as ppdg_config_parser
import model_inversion.ppdg.attack as ppdg
import model_inversion.ppdg.modify_to_repo as ppdg_modify

from utils import wandb_helpers, load_trained_models
from utils.lambdalabs.scripting import terminate_lambdalabs_instance

@hydra.main(config_path="configuration/model_inversion", config_name="config.yaml", version_base="1.3")
def run_model_inversion(attack_config):

    if torch.cuda.is_available() and attack_config.training.device != "cuda":
        question = f"\nCuda is available but not configured from command to be used! Do you wish to use cuda instead of {attack_config.training.device}?\nType y to use cuda, enter if not:"
        
        use_cuda = input(question)
        
        if use_cuda.lower().strip() == "y":
            attack_config.training.device = 'cuda'


    if attack_config.training.wandb.track:
        wandb_run = wandb_helpers.wandb_init(attack_config)
    else:
        wandb_run = None

    try:
        load_best_model = attack_config.training.load_best_model
    except omegaconf.errors.ConfigAttributeError:
        load_best_model = True
        print("Warning: load_best_model not found in config, defaulting to True")

    target_config, target_weights_path = load_trained_models.get_target_config_and_weights(attack_config, load_best_model=load_best_model)

    if attack_config.training.wandb.track:
        wandb.log({
            "defense": target_config.defense.name,
            "architecture": target_config.model.architecture,
            "dataset": target_config.dataset.dataset,
        })
        
    # Load data
    transform = get_transforms(target_config, train=False)
    
    train_dataset, _, _ = get_datasets(config=target_config,
                                                            train_transform=transform,
                                                            test_transform=None)
    
    train_loader, val_loader, test_loader = get_data_loaders(target_config)
    
    target_model = get_model(target_config)

    # Load defense
    target_model = get_defense(config=target_config, model=target_model)

    # Load model weights
    target_model.load_model(target_weights_path)

    if attack_config.training.calculate_train_accuracy:
        train_loss, train_accuracy = target_model.evaluate(train_loader)
        print("train_loss", train_loss)
        print("train_accuracy", train_accuracy)
    test_loss, test_accuracy = target_model.evaluate(test_loader)
    print("test_loss", test_loss)
    print("test_accuracy", test_accuracy)

    if attack_config.training.wandb.track:
        if attack_config.training.calculate_train_accuracy:
            wandb.log({"train_loss": train_loss})
            wandb.log({"train_accuracy": train_accuracy})
        wandb.log({"test_loss": test_loss})
        wandb.log({"test_accuracy": test_accuracy})

    if attack_config.attack.name == "plug_and_play":
        
        evaluation_config, evaluation_weights_path = load_trained_models.get_evaluation_config_and_weights(attack_config)
        
        # Load evaluation model
        evaluation_model = get_model(evaluation_config)
        evaluation_model.load_model(evaluation_weights_path)
        
        # Convert to plug and play compatibility 
        target_model = ppa_modify.model_compatibility_wrapper(model = target_model, target_config = target_config)
        evaluation_model = ppa_modify.model_compatibility_wrapper(model = evaluation_model, target_config = evaluation_config)
        
        new_attack_config_path = ppa_modify.convert_configs(target_config, attack_config)
        new_attack_config = ppa_config_parser.AttackConfigParser(new_attack_config_path)
        
        target_model.eval()
        evaluation_model.eval()
        
        ppa.attack(
            config = new_attack_config,
            target_dataset = train_dataset,
            target_model = target_model,
            evaluation_model = evaluation_model,
            target_config = target_config,
            wandb_run = wandb_run,
            )
   
    if attack_config.attack.name == "IF-GMI":
        evaluation_config, evaluation_weights_path = load_trained_models.get_evaluation_config_and_weights(attack_config)
        
        # Load evaluation model
        evaluation_model = get_model(evaluation_config)
        evaluation_model.load_model(evaluation_weights_path)
        
        # Convert to plug and play compatibility (same wrapper for PPA and IF-GMI)
        target_model = ppa_modify.model_compatibility_wrapper(model = target_model, target_config = target_config)
        evaluation_model = ppa_modify.model_compatibility_wrapper(model = evaluation_model, target_config = evaluation_config)
        
        new_attack_config_path = if_gmi_modify.convert_configs(target_config, attack_config)
        new_attack_config = if_gmi_config_parser.AttackConfigParser(new_attack_config_path)
        
        target_model.eval()
        evaluation_model.eval()
        
        if_gmi.attack(
            config = new_attack_config,
            target_dataset = train_dataset,
            target_model = target_model,
            evaluation_model = evaluation_model,
            target_config = target_config,
            wandb_run = wandb_run)
        
    if attack_config.attack.name == "PPDG":
        evaluation_config, evaluation_weights_path = load_trained_models.get_evaluation_config_and_weights(attack_config)

        # Load evaluation model
        evaluation_model = get_model(evaluation_config)
        evaluation_model.load_model(evaluation_weights_path)
        
        target_model = ppa_modify.model_compatibility_wrapper(model = target_model, target_config = target_config)
        evaluation_model = ppa_modify.model_compatibility_wrapper(model = evaluation_model, target_config = evaluation_config)

        new_attack_config_path = ppdg_modify.convert_configs(target_config, attack_config)
        new_attack_config = ppdg_config_parser.AttackConfigParser(new_attack_config_path)
        
        target_model.eval()
        evaluation_model.eval()
        
        ppdg.attack(
            config = new_attack_config,
            target_dataset = train_dataset,
            target_model = target_model,
            evaluation_model = evaluation_model,
            target_config = target_config,
            wandb_run = wandb_run)
        
    print("done")

    if attack_config.LL_terminate_on_end:
        if attack_config.LL_sleep_before_terminate:
            print("Sleeping before LL termination... ")
            time.sleep(int(attack_config.LL_sleep_before_terminate))
        print("Terminating current Lambda Labs instance... ")
        terminate_lambdalabs_instance()
    
if __name__ == "__main__":
    run_model_inversion()
