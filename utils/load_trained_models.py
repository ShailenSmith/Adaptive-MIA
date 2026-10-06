import sys
from omegaconf import OmegaConf

from utils import wandb_helpers

def get_target_config_and_weights(attack_config, load_best_model=True):
    ## Get target model config and weights
    # If targeting a model trained with wandb tracking
    if attack_config.target_wandb_id:
        target_config, run_name = wandb_helpers.get_config(entity   = attack_config.training.wandb.entity,
                                                           project  = attack_config.training.wandb.target_project,
                                                           run_id = attack_config.target_wandb_id)
        
        target_weights_path = wandb_helpers.get_weights(
                                                        entity   = attack_config.training.wandb.entity,
                                                        project  = attack_config.training.wandb.target_project,
                                                        run_id = attack_config.target_wandb_id,
                                                        save_as  = target_config.training.save_as,
                                                        load_best_model=load_best_model) # This is target config to get the right file name
        
        # if not load_best_model: # downloads the last model weights as well
        #     last_target_weights_path = wandb_helpers.get_weights(
        #                                                         entity   = attack_config.training.wandb.entity,
        #                                                         project  = attack_config.training.wandb.target_project,
        #                                                         run_id = attack_config.target_wandb_id,
        #                                                         save_as  = target_config.training.save_as,
        #                                                         load_best_model=False)


    # If targeting a local runs
    elif attack_config.target_config_path:
        try:
            target_config = OmegaConf.load(attack_config.target_config_path)
            run_name = target_config.training.save_as
            print("Configuration loaded successfully.")
            
        except Exception as e:
            print(f"Error loading YAML file: {e}", file=sys.stderr)
            raise
    
        target_weights_path = f"classifiers/saved_models/{run_name}.pth"
        
    else:
        raise ValueError("Please provide either a wandb id or a path to a configuration file")
    
    return target_config, target_weights_path


def get_evaluation_config_and_weights(attack_config):
    ## Get evaluation model weights and config
    if attack_config.attack.evaluation_model.wandb_id:
        evaluation_config, run_name = wandb_helpers.get_config(
                                                                entity   = attack_config.training.wandb.entity,
                                                                project  = attack_config.training.wandb.evaluation_project,
                                                                run_id = attack_config.attack.evaluation_model.wandb_id,
                                                                )
                    
        evaluation_weights_path = wandb_helpers.get_weights(
                                                            entity   = evaluation_config.training.wandb.entity,
                                                            project  = evaluation_config.training.wandb.project,
                                                            run_id = attack_config.attack.evaluation_model.wandb_id,
                                                            save_as  = evaluation_config.training.save_as)
        
    # If using a local run
    elif attack_config.attack.evaluation_model.config_path:
        try:
            evaluation_config = OmegaConf.load(attack_config.attack.evaluation_model.config_path)
            run_name = evaluation_config.training.save_as
            print("Configuration loaded successfully.")
            
        except Exception as e:
            print(f"Error loading YAML file: {e}", file=sys.stderr)
            raise
    
        evaluation_weights_path = f"classifiers/saved_models/{run_name}.pth"
        
    else:
        raise ValueError("Please provide either a wandb id or a path to a configuration file")
    
    return evaluation_config, evaluation_weights_path