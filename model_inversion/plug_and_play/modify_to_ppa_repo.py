import torch
import torch.nn as nn
import numpy as np
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
    new_dict['wandb'] = {}
    new_dict['wandb']['enable_logging'] = attack_config.training.wandb.track
    new_dict['wandb']['wandb_init_args'] = {}
    new_dict['wandb']['wandb_init_args']['project'] = attack_config.training.wandb.project
    new_dict['wandb']['wandb_init_args']['save_code'] = True #? Don't know what they use this for
    
    save_path = "Plug_and_Play_Attacks/configs/attacking/"
    save_as = save_path + attack_config.training.save_as
    
    with open(f"{save_as}.yaml", "w") as file:
        yaml.dump(new_dict, file, default_flow_style=None, sort_keys=False)
    
    # config_obj = AttackConfigParser()
    
    return save_as + ".yaml"

class model_compatibility_wrapper(nn.Module):
    def __init__(self, model, target_config):
        super(model_compatibility_wrapper, self).__init__()
        
        #   # Flattening the model hierarchy by copying attributes directly
        # for name, module in model.named_children():
        #     self.add_module(name, module)

        
        self._model = model # Makes it possible to access the abstract classifier when defining functions in the class
        
        self.name = target_config.model.name
        self.device = torch.device(target_config.training.device)
        self.num_classes = target_config.dataset.n_classes
        self.to(self.device)

        self.use_cuda = torch.cuda.is_available() and target_config.training.device == "cuda"
        
    @property
    def model(self):
        """Property to simplify access to the underlying model."""
        return self._model.model if hasattr(self._model, "model") else self._model
        
    def forward(self, x):
        if getattr(self._model, "forward_only_logits", None): # some models have forward calls that return multiple outputs, we just want the logits
            return self._model.forward_only_logits(x)
        else:
            return self._model.forward(x)

    def fit(self, train_loader, val_loader):
        return self._model.train_model(train_loader, val_loader)

    def evaluate(self, data_loader):
        avg_loss, accuracy = self._model.evaluate(data_loader)
        return accuracy, avg_loss

    def predict(self, x):
        return self._model.predict(x)
    
    def set_parameter_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def count_parameters(self, only_trainable=False):
        if only_trainable:
            return sum(param.numel() for param in self.parameters()
                    if param.requires_grad)
        return sum(param.numel() for param in self.parameters())

    def __str__(self):
        num_params = np.sum([param.numel() for param in self.parameters()])
        if self.name:
            return self.name + '\n' + super().__str__(
            ) + f'\n Total number of parameters: {num_params}'
        else:
            return super().__str__(
            ) + f'\n Total number of parameters: {num_params}'
    
    





