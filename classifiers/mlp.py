# smaller model for testing
import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier

class FullyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(FullyConnectedBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
    def forward(self, x):
        return self.block(x)

class MLP(AbstractClassifier, nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__(config)
        # self.config = config
        # self.feature_extractor, self.classification_layer = self.init_model()
        
    def init_model(self):
        super(MLP, self).init_model() # AbstractClassifier's init_model will return an ElementwiseLinear layer if drop_layer=True inself.config
        
        # Input features based on the dataset
        in_features =self.config.dataset.input_size[1] *self.config.dataset.input_size[2] *self.config.dataset.input_size[0]
        
        first_layer = FullyConnectedBlock(in_features,self.config.model.hyper.n_neurons,self.config.model.hyper.dropout)
        
        middle_layers = [FullyConnectedBlock(self.config.model.hyper.n_neurons,self.config.model.hyper.n_neurons,self.config.model.hyper.dropout) for _ in range(self.config.model.hyper.n_depth)]
        
        last_layer = nn.Linear(self.config.model.hyper.n_neurons,self.config.dataset.n_classes)

        modules = [first_layer,
                   *middle_layers,
                   ]

        feature_extractor = nn.Sequential(*modules)
        classification_layer = last_layer

        return feature_extractor, classification_layer

