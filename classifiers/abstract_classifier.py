import torch
import torch.nn as nn
import numpy as np
import wandb
import os
from tqdm import tqdm
import omegaconf

from data_processing.data_augmentation import get_batch_transforms

class AbstractClassifier(nn.Module):
    """ 
    This is an abstract class for the classifiers. It contains the train, forward, predict, save_model, load_model methods, among others. It should not be used directly. 
    """
    
    def __init__(self, config):
        super(AbstractClassifier, self).__init__()
        self.device = config.training.device
        self.config = config

        if config.model.criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
            self.criterionSum = nn.CrossEntropyLoss(reduction='sum')

        if config.model.criterion == "MSE":
            self.criterion = nn.MSELoss()
            self.criterionSum = nn.MSELoss(reduction='sum')

        self.batch_transforms = get_batch_transforms(config)

        self.feature_extractor, self.classification_layer = self.init_model()

    def init_model(self): # replace this in child classes
        feature_extractor = None
        classification_layer = None
        return feature_extractor, classification_layer
    
    def train_one_epoch(self, train_loader):
        
        self.train()
        total_loss = 0
        loss_calculated = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            if self.batch_transforms: # apply cutmix or mixup if specified in config
                data, target = self.batch_transforms(data, target)
            
            self.optimizer.zero_grad()
            output = self(data)

            loss = self.get_loss(output, target)

            total_loss += loss.item()          

            loss_calculated += len(data)

            if self.config.training.verbose == 2:
                print("loss: ", loss.item())

            loss.backward()
            self.optimizer.step()
            self.train_step += 1

        train_loss = total_loss / loss_calculated
        return train_loss

    
    def train_model(self, train_loader, val_loader):

        if self.config.training.save_as:
            self.save_as = self.config.training.save_as + ".pth"
        elif self.config.training.wandb.track:
            self.save_as = wandb.run.id + ".pth" 
        else:
            raise ValueError("Please provide a name to save the model when not using wandb tracking")
        
        # Distribute model on GPU's
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.feature_extractor = nn.DataParallel(self.feature_extractor)
            self.classification_layer = nn.DataParallel(self.classification_layer)
        
        if self.config.model.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), # weight_decay=0 by default
                                              lr=self.config.model.hyper.lr,
                                              betas=(0.9, self.config.model.hyper.beta2),
                                              )
        else:
            raise ValueError(f"Optimizer {self.config.model.optimizer} not recognized")

        
        if not self.config.model.lr_scheduler:
            self.lr_scheduler = None
        
        elif self.config.model.lr_scheduler == "MultiStepLR": 
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                             milestones=self.config.model.hyper.milestones,
                                                             gamma=self.config.model.hyper.gamma,
                                                             )
        else:
            raise ValueError(f"Learning rate scheduler {self.config.model.lr_scheduler} not recognized")

        self.train_step = 0
        best_loss = np.inf
        no_improve_epochs = 0

        self.to(self.device)
        print("\nTraining using ", self.device)
        
        for epoch in tqdm(range(self.config.model.hyper.epochs), desc="Training", total=self.config.model.hyper.epochs):         
            train_loss = self.train_one_epoch(train_loader)

            if self.config.training.verbose:
                print(f'Epoch: {epoch + 1}') 
                print(f"Train loss: {train_loss}")
            
            if self.config.training.wandb.track:
                wandb.log({"train_loss": train_loss, "train_step": self.train_step, "epoch": epoch+1})
                if self.config.model.lr_scheduler:
                    wandb.log({"learning_rate" : self.lr_scheduler.get_last_lr()[0], "train_step" : self.train_step, "epoch": epoch+1})

            if val_loader and epoch % self.config.training.evaluate_freq == 0:
                val_loss, val_accuracy = self.evaluate(val_loader)

                _, train_accuracy = self.evaluate(train_loader)
                del _
                
                if self.config.training.verbose:
                    print(f'Train Accuracy: {train_accuracy}')
                    print(f'Validation loss: {val_loss}') 
                    print(f'Val Accuracy: {val_accuracy}')
                
                if self.config.training.wandb.track:
                    wandb.log({"train_accuracy": train_accuracy, "train_step": self.train_step, "epoch": epoch+1})
                    wandb.log({"val_loss": val_loss, "train_step": self.train_step, "epoch": epoch+1})
                    wandb.log({"val_accuracy": val_accuracy, "train_step": self.train_step, "epoch": epoch+1})
                
                # early stopping criteria
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model(self.save_as)
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= self.config.model.hyper.patience:
                        print("Early stopping")
                        break

                if self.config.training.save_many_accs:
                    thresh_list = self.config.training.early_stopping_accuracy_thresholds

                    for thresh in thresh_list:
                        if val_accuracy >= (thresh-0.0025) and val_accuracy < (thresh+0.0025):
                            file_path = self.save_as
                            base, ext = os.path.splitext(file_path)
                            save_path = f"{base}_acc{thresh}{ext}"
                            self.save_model(save_path)
                            print(f">>> Saved model with accuracy threshold {thresh} at {save_path}")

            if self.config.model.lr_scheduler:
                self.lr_scheduler.step()
        
        file_path = self.save_as
        base, ext = os.path.splitext(file_path)
        last_epoch_path = f"{base}_last_epoch{ext}"
        
        self.save_model(last_epoch_path)
        
        # Load the best model
        self.load_model(f"classifiers/saved_models/{self.save_as}", map_location=self.device)
    
    def evaluate(self, loader):
        self.to(self.device)
        self.eval()
        
        correct = 0
        total_loss = 0
        total_instances = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)

                loss = self.get_loss(output, target)

                total_loss += loss.item()
                total_instances += len(data)
                
                pred = torch.argmax(output, dim=1)
                correct += (pred == target).sum().item()
        
        avg_loss = total_loss / total_instances
        accuracy = correct / total_instances
        
        return avg_loss, accuracy


    def get_loss(self, output, target): # calculate loss 
        loss = self.criterionSum(output, target)
        return loss


    def forward(self, x):
        z = self.feature_extractor(x)
        logits = self.classification_layer(z)
        return logits


    def embed_img(self, x): # gets the embedding of the image pre-classification
        return self.feature_extractor(x)


    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)


    def save_model(self, name):
        path = f"classifiers/saved_models/{name}"
        if isinstance(self.feature_extractor, nn.DataParallel): # self.feature_extractor is on DataParallel, need to only save self.feature_extractor.module
            state = {
                    "feature_extractor": self.feature_extractor.module.state_dict(),
                    "classification_layer": self.classification_layer.module.state_dict(),
                }
        else:
            state= {
                "feature_extractor": self.feature_extractor.state_dict(),
                "classification_layer": self.classification_layer.state_dict(),
            }
        torch.save(state, path)

        
    def load_model(self, file_path, map_location = None):
        
        if map_location is None:
            state = torch.load(file_path, weights_only=True)
        else:
            state = torch.load(file_path, map_location=map_location, weights_only=True)

            
        if 'model.fc.weight' in state.keys(): # fix old state dicts so that they match new AbstractClassifier format
            new_state = {}
            for key in state.keys():
                if not key.startswith('feature_extractor.'): # don't include 'feature_extractor.' ones
                    new_key = key.replace('model.', '')
                    new_state[new_key] = state[key]
            classification_layer_state = {}
            classification_layer_state['weight'] = new_state['fc.weight']
            classification_layer_state['bias'] = new_state['fc.bias']
            del new_state['fc.weight']
            del new_state['fc.bias']
            state = {}
            state['feature_extractor'] = new_state.copy()
            state['classification_layer'] = classification_layer_state
        elif 'model' in state.keys(): # fix old state dicts so that they match new AbstractClassifier format
            classification_layer_state = {}
            classification_layer_state['weight'] = state['model']['fc.weight']
            classification_layer_state['bias'] = state['model']['fc.bias']
            del state['model']['fc.weight']
            del state['model']['fc.bias']
            state['feature_extractor'] = state.pop('model')
            state['classification_layer'] = classification_layer_state


        if state['classification_layer']['weight'].shape[0] != self.config.dataset.n_classes: # needed for backcompatibility with loaded Inception models
            state['classification_layer']['weight'] = state['classification_layer']['weight'][:self.config.dataset.n_classes]
            state['classification_layer']['bias'] = state['classification_layer']['bias'][:self.config.dataset.n_classes]

        
        if isinstance(self.feature_extractor, nn.DataParallel):
            self.feature_extractor.module.load_state_dict(state['feature_extractor'])
            self.classification_layer.module.load_state_dict(state['classification_layer'])
        else:
            self.feature_extractor.load_state_dict(state['feature_extractor'])
            self.classification_layer.load_state_dict(state['classification_layer'])