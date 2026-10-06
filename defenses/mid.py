import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

import wandb

from classifiers.abstract_classifier import AbstractClassifier

def apply_MID_defense(config, model:AbstractClassifier):
    
    class MID(model.__class__):
            
        def __init__(self, config):
            super(MID, self).__init__(config)            
            # list_of_layers = list(self.feature_extractor.children())
            last_layer = self.classification_layer
            # self.feat_dim = 512 * 2 * 2
            self.feat_dim = last_layer.in_features
            self.k = self.feat_dim // 2
            self.n_classes = config.dataset.n_classes

            self.only_logits = False
            
            # # remove final classification layer
            # if hasattr(self.model, "fc"):
            #     self.model.fc = nn.Identity()
            # elif hasattr(self.model, "classifier"):
            #     self.model.classifier = nn.Identity()
            # elif config.model.name == "CNN" or config.model.name == "MLP":
            #     print("Warning: Defense has not been tested with the 'CNN' and 'MLP' models")
            #     self.model = nn.Sequential(*list(self.model.children())[:-1])
            # else:  
            #     raise ValueError("Cannot recognize the classification layer of the model. Make sure the last layer is named 'fc' or 'classifier'")
            
            self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
            self.classification_layer = nn.Linear(self.k, self.n_classes)
            
        def forward(self, x, mode="train"):
            feature = self.feature_extractor(x)

            feature = feature.view(feature.size(0), -1)
            
            statis = self.st_layer(feature)
            
            mu, std = statis[:, :self.k], statis[:, self.k:]
            
            std = F.softplus(std-5, beta=1) #! Parameters?
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
            res = mu + std * eps
            out = self.classification_layer(res)
            
            # __, iden = torch.max(out, dim=1)
            # iden = iden.view(-1, 1)

            if self.only_logits:
                return out
            else:
                return [feature, mu, std, out]

        def forward_only_logits(self, x):
            ___, ___, ___, out = self(x)
            return out
        
        def save_model(self, name):
            path = f"classifiers/saved_models/{name}"
            if isinstance(self.feature_extractor, nn.DataParallel): # self.feature_extractor, self.st_layer, and self.classification_layer are on DataParallel
                state = {
                    "feature_extractor": self.feature_extractor.module.state_dict(),
                    "st_layer": self.st_layer.module.state_dict(),
                    "classification_layer": self.classification_layer.module.state_dict(),
                }
            else:
                state= {
                    "feature_extractor": self.feature_extractor.state_dict(),
                    "st_layer": self.st_layer.state_dict(),
                    "classification_layer": self.classification_layer.state_dict(),
                }
            torch.save(state, path)

        def load_model(self, file_path, map_location = None):
            if map_location is None:
                state = torch.load(file_path, weights_only=True)
            else:
                state = torch.load(file_path, map_location=map_location, weights_only=True)  

            if isinstance(self.feature_extractor, nn.DataParallel):
                self.feature_extractor.module.load_state_dict(state['feature_extractor'])
                self.st_layer.module.load_state_dict(state['st_layer'])
                self.classification_layer.module.load_state_dict(state['classification_layer'])
            else:
                self.feature_extractor.load_state_dict(state['feature_extractor'])
                self.st_layer.load_state_dict(state['st_layer'])
                self.classification_layer.load_state_dict(state['classification_layer'])

        def predict(self, x):
            feature, mu, std, out = self(x)
            
            out = torch.argmax(out, dim=1)
        
            return out

        def train_model(self, train_loader, val_loader):
            if torch.cuda.device_count() > 1:
                self.st_layer = nn.DataParallel(self.st_layer) # self.feature_extractor will be put on DataParallel in super train_model call, so we just need to put the st_layer on DataParallel here

            return super(MID, self).train_model(train_loader, val_loader)
        
        def get_loss(self, output, target):
            ___, mu, std, out_prob = output
            cross_loss = self.criterionSum(out_prob, target)
            info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).sum()
            if hasattr(self, "train_step"):
                if self.config.training.wandb.track and self.train_step % 50 == 0:
                    if self.train_step >= 50:
                        wandb.log({"info_loss": info_loss.item()}) # log info_loss value to wandb every 50 steps

            loss = cross_loss + self.config.defense.beta * info_loss
    
            return loss
        
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
                    
                    _, _, _, out = output #! This is the change
                    
                    pred = torch.argmax(out, dim=1)
                    correct += (pred == target).sum().item()
            
            avg_loss = total_loss / total_instances
            accuracy = correct / total_instances
            
            return avg_loss, accuracy
            
    protected_model = MID(config)
    return protected_model
