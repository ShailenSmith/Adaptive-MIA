# smaller model for testing
import torch
import torch.nn as nn
from classifiers.abstract_classifier import AbstractClassifier
import torchvision
from torchvision import transforms

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.block = nn.Sequential(
            self.conv,
            self.bn,
            self.relu,
            self.pool,
                )
        
    def forward(self, x):
        return self.block(x)

class CNN(AbstractClassifier, nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__(config)
        # self.config = config
        # self.feature_extractor, self.classification_layer = self.init_model()


    def init_model(self):
        super(CNN, self).init_model()
        self.n_channels = self.config.dataset.input_size[0]
        
        first_conv = ConvBlock(self.n_channels, self.config.model.hyper.n_neurons, self.config.model.hyper.kernel_size, self.config.model.hyper.stride)
        
        conv_layers = [ConvBlock(self.config.model.hyper.n_neurons * 2**i, self.config.model.hyper.n_neurons * 2**(i+1), 
                                self.config.model.hyper.kernel_size, self.config.model.hyper.stride) for i in range(self.config.model.hyper.n_depth)]
        
        # Calculate the output size after the convolutional layers
        input_height, input_width = self.config.dataset.input_size[1], self.config.dataset.input_size[2]
        num_layers = self.config.model.hyper.n_depth + 1  # Including the first conv layer
        
        # Keep track of the output size after each layer
        for _ in range(num_layers):
            input_height = (input_height - 2) // 2 + 1
            input_width = (input_width - 2) // 2 + 1
        
        # Calculate the output size after the convolutional layers
        conv_output_size = self.config.model.hyper.n_neurons * 2**self.config.model.hyper.n_depth * input_height * input_width

        feature_extractor = nn.Sequential(
            first_conv,
            *conv_layers,
            nn.Flatten(),
            nn.Linear(conv_output_size , self.config.model.hyper.linear_output_size), 
            nn.ReLU(),
            nn.Dropout(self.config.model.hyper.dropout),
            )

        classification_layer = nn.Linear(self.config.model.hyper.linear_output_size, self.config.dataset.n_classes)

        return feature_extractor, classification_layer
    
