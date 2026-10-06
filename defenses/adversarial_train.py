import torch
import torch.nn as nn
import wandb

from classifiers.get_model import get_model
from classifiers.abstract_classifier import AbstractClassifier
from utils import wandb_helpers
from utils.adversarial_attacks import PGDAttack

def apply_adversarial_training_defense(config, model:AbstractClassifier):
    
    class AdversarialTraining(model.__class__): 
        
        def __init__(self, config):
            super(AdversarialTraining, self).__init__(config)
            
            try:
                self.method = self.config.defense.method
            except AttributeError:
                print("Warning: defense.method not set. Using PGD by default.")
                self.method = "pgd"

            try:
                self.targeted = self.config.defense.targeted
            except AttributeError:
                print("Warning: defense.targeted not set. Using untargeted by default.")
                self.targeted = False
            
            try:
                use_helper_model = self.config.defense.helper_model.use
            except AttributeError:
                print("Warning: defense.helper_model.use not set. Using no helper model by default.")
                use_helper_model = False
            
            ### Setup the helper model if needed
            if use_helper_model:
                target_model = self.load_helper_model()
                print(f"Using helper model: {self.config.defense.helper_model.wandb.run_id}")
            else:
                target_model = self
                
            ### Setup the attack
            if self.method == "pgd":
                self.attack = PGDAttack(
                    model=target_model,
                    loss_fn=nn.CrossEntropyLoss(),
                    epsilon=config.defense.pgd.epsilon,
                    step_size=config.defense.pgd.step_size,
                    iterations=config.defense.pgd.iterations,
                    targeted=self.targeted,
                    random_start=config.defense.pgd.random_start,
                    min_clamp=-1,
                    max_clamp=1
                )
            elif self.method == "fgsm":
                self.attack = FGSM(
                    model=target_model,
                    loss_fn=nn.CrossEntropyLoss(),
                    epsilon=config.defense.fgsm.epsilon,
                    targeted=self.targeted,
                    min_clamp=-1,
                    max_clamp=1
                )
            else:
                raise ValueError(f"Unknown attack method: {self.method}")
            
        


        def train_one_epoch(self, train_loader):
            self.train()
            total_loss = 0
            loss_calculated = 0
            
            attack_success_sum = 0
            total_loss_normal = 0
            total_loss_adversarial = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                
                if self.config.defense.loss_weighting.standard:
                    output = self(data)
                    loss_normal = self.get_loss(output, target)
                else:
                    loss_normal = 0
                
                if self.targeted:
                    # choose another target randomly
                    attack_target = torch.randint(0, self.config.dataset.n_classes, (target.shape[0],)).to(self.device)
                    
                    # if the target is the same as the original label, choose another one
                    same_target = (attack_target == target)
                    attack_target[same_target] = (attack_target[same_target] + 1) % self.config.dataset.n_classes
                else:
                    attack_target = target    
                
                data_adv = self.attack(data, attack_target)

                output_adv = self(data_adv)

                logits_adv = output_adv[0] if isinstance(output_adv, tuple) else output_adv
                pred_adv = torch.argmax(logits_adv, dim=1)
                
                if self.targeted:
                    # For targeted attacks, we want to see if the model is fooled into predicting the target class
                    attack_success_sum += (pred_adv == attack_target).sum()
                else:
                    # For untargeted attacks, we want to see if the model is fooled into predicting a different class
                    attack_success_sum += (pred_adv != target).sum()
                
                if self.config.defense.use_adversarial_label: #! Maybe not needed
                    target = pred_adv
                    
                if self.targeted:
                    loss_adv = self.get_loss(output_adv, attack_target)
                else:
                    loss_adv = self.get_loss(output_adv, target)
                
                if self.config.defense.loss_weighting.adversarial_log_scale:
                    loss_adv = torch.log(loss_adv + 1e-8)
                
                loss = self.config.defense.loss_weighting.standard * loss_normal + self.config.defense.loss_weighting.adversarial * loss_adv

                if loss_normal != 0:
                    total_loss_normal += loss_normal.item()
                    
                total_loss_adversarial += loss_adv.item()
                total_loss += loss.item()          

                loss_calculated += len(data)

                if self.config.training.verbose == 2:
                    print("loss: ", loss.item())

                loss.backward()
                self.optimizer.step()
                self.train_step += 1

            train_loss = total_loss / loss_calculated
            
            train_loss_normal = total_loss_normal / loss_calculated
            train_loss_adversarial = total_loss_adversarial / loss_calculated
            
            attack_success_percentage = attack_success_sum / loss_calculated
            
            print(f"(normal: {train_loss_normal:.4f}, adversarial: {train_loss_adversarial:.4f}), attack_success_percentage: {attack_success_percentage:.4f}")
            
            if wandb.run is not None:
                wandb.log({"train_step": self.train_step,
                           "train_loss_adversarial": train_loss_adversarial, 
                           "train_loss_normal": train_loss_normal, 
                           "attack_success_percentage": attack_success_percentage})
            
            return train_loss
        
        def load_helper_model(self):
            helper_weights_path = wandb_helpers.get_weights(
                                entity   = self.config.defense.helper_model.wandb.entity,
                                project  = self.config.defense.helper_model.wandb.project,
                                run_id   = self.config.defense.helper_model.wandb.run_id
            )
             
            helper_config, helper_run_name = wandb_helpers.get_config(
                                entity   = self.config.defense.helper_model.wandb.entity,
                                project  = self.config.defense.helper_model.wandb.project,
                                run_id   = self.config.defense.helper_model.wandb.run_id
            )

            helper_model = get_model(helper_config)
            helper_model.load_model(helper_weights_path)
            
            # Don't track gradients
            for param in helper_model.parameters():
                param.requires_grad = False
                
            helper_model.eval()
            helper_model.to(self.device)
            
            return helper_model
        

    advesarial_defended_model = AdversarialTraining(config)
    
    return advesarial_defended_model

# Inspired by: https://github.com/Harry24k/FGSM-pytorch/blob/master/FGSM.ipynb

class FGSM:
    def __init__(self, model, loss_fn, epsilon:float, targeted:bool, min_clamp=-1, max_clamp=1):
        self.model = model
        self.device = model.device
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.targeted = targeted
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp

    def __call__(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        images.requires_grad = True

        # Generate in eval mode: loss_fn is a plain tensor loss and can't handle
        # tuple outputs (e.g. Inception's (logits, aux_logits) in train mode), and
        # eval mode also avoids polluting BatchNorm running stats with adversarial images.
        was_training = self.model.training
        self.model.eval()
        try:
            outputs = self.model(images)

            self.model.zero_grad()

            if self.targeted:
                cost = -self.loss_fn(outputs, labels).to(self.model.device)
            else:
                cost = self.loss_fn(outputs, labels).to(self.model.device)

            cost.backward()
        finally:
            if was_training:
                self.model.train()

        attack_images = images + self.epsilon * images.grad.sign()
        attack_images = torch.clamp(attack_images, self.min_clamp, self.max_clamp)

        attack_images = attack_images.detach()
        attack_images.requires_grad = False

        return attack_images