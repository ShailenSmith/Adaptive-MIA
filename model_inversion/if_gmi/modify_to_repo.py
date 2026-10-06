
import yaml

from omegaconf import OmegaConf

def convert_configs(target_config, attack_config):
    new_dict = {}

    new_dict.update(OmegaConf.to_container(attack_config.attack, resolve = True))
    
    new_dict["dataset"] = target_config.dataset.dataset
    new_dict["seed"] = attack_config.training.seed
    new_dict['evaluation_model']['num_classes'] = target_config.dataset.n_classes
    new_dict['candidates']['candidate_search']['resize'] = target_config.dataset.input_size[1]
    
    # Convert wandb configuration
    # new_dict['wandb'] = {}
    # new_dict['wandb']['enable_logging'] = attack_config.training.wandb.track
    # new_dict['wandb']['wandb_init_args'] = {}
    # new_dict['wandb']['wandb_init_args']['project'] = attack_config.training.wandb.project
    # new_dict['wandb']['wandb_init_args']['save_code'] = True #? Don't know what they use this for
    
    # new_dict['wandb_log'] = attack_config.training.wandb.track
    
    save_path = "model_inversion/if_gmi/configs/"
    save_as = save_path + attack_config.training.save_as
    
    with open(f"{save_as}.yaml", "w") as file:
        yaml.dump(new_dict, file, default_flow_style=None, sort_keys=False)
    
    # config_obj = AttackConfigParser()
    
    return save_as + ".yaml"