from classifiers.abstract_classifier import AbstractClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

def apply_gaussian_noise_defense(config, model:AbstractClassifier):
    
    class GaussianNoise(model.__class__):
        
        def __init__(self, config):
            super(GaussianNoise, self).__init__(config)

            # # remove final classification layer
            # if hasattr(self.model, "fc"):
            #     self.fc_layer = self.model.fc
            #     self.model.fc = nn.Identity()
            # elif hasattr(self.model, "classifier"):
            #     self.fc_layer = self.model.classifier
            #     self.model.classifier = nn.Identity()
            # elif config.model.name == "CNN" or config.model.name == "MLP":
            #     print("Warning: Defense has not been tested with the 'CNN' and 'MLP' models")
            #     self.fc_layer = list(self.model.children())[-1]
            #     self.model = nn.Sequential(*list(self.model.children())[:-1])
            # else:  
            #     raise ValueError("Cannot recognize the classification layer of the model. If not using 'CNN' or 'MLP', \
            #         make sure the classification layer is named either 'fc' or 'classifier'")
            
            self.noise_layer = NoiseLayer(shape=self.classification_layer.in_features, stddev=0, seed=self.config.training.seed) # no noise to start

        def forward(self, x):
            z = self.feature_extractor(x)
            z_noised = self.noise_layer(z)
            logits = self.classification_layer(z_noised)
            return logits

        def train_model(self, train_loader, val_loader):
            if torch.cuda.device_count() > 1:
                self.noise_layer = nn.DataParallel(self.noise_layer)
            return super(GaussianNoise, self).train_model(train_loader, val_loader)

        def save_model(self, name):
            path = f"classifiers/saved_models/{name}"

            current_noise_layer = self.get_noise_layer()
            if torch.std(current_noise_layer.noise) == 0:
                # add noise to saved model
                new_noise_layer = NoiseLayer(shape=current_noise_layer.noise.shape,
                                            stddev=self.config.defense.stddev,
                                            seed=self.config.training.seed)
            else:
                new_noise_layer = current_noise_layer

            if isinstance(self.feature_extractor, nn.DataParallel): # self.feature_extractor, self.noise_layer, and self.classification_layer are on DataParallel

                state = {
                    "model": self.feature_extractor.module.state_dict(),
                    "noise_layer": new_noise_layer.state_dict(),
                    "classification_layer": self.classification_layer.module.state_dict(),
                }
            else:
                state= {
                    "model": self.feature_extractor.state_dict(),
                    "noise_layer": new_noise_layer.state_dict(),
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
                self.noise_layer.module.load_state_dict(state['noise_layer'])
                self.classification_layer.module.load_state_dict(state['classification_layer'])
            else:
                self.feature_extractor.load_state_dict(state['feature_extractor'])
                self.noise_layer.load_state_dict(state['noise_layer'])
                self.classification_layer.load_state_dict(state['classification_layer'])

        def get_noise_layer(self):
            if isinstance(self.noise_layer, nn.DataParallel):
                return self.noise_layer.module
            else:
                return self.noise_layer

    gaussian_noise_defended_model = GaussianNoise(config)
    return gaussian_noise_defended_model


class NoiseLayer(nn.Module):
    def __init__(self, shape, stddev, seed=42):
        super(NoiseLayer, self).__init__()

        torch.manual_seed(seed)
        self.register_buffer("noise", torch.randn(shape) * stddev)

    def forward(self, x):
        return x + self.noise


