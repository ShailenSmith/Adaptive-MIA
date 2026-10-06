from classifiers.abstract_classifier import AbstractClassifier
import torch.nn as nn

# Implement your custom classifier here, look at cnn.py or mlp.py for examples

class CustomClassifier(AbstractClassifier, nn.Module):
    
    def __init__(self, config):
        super(CustomClassifier, self).__init__(config)
        self.config = config
        self.feature_extractor, self.classification_layer = self.init_model()

    def init_model(self):
        raise NotImplementedError("Implement the init_model method in your custom classifier")