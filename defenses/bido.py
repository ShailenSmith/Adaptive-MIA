from argparse import ArgumentParser
import wandb

import torch
import torch.nn as nn
import torchvision

from classifiers.abstract_classifier import AbstractClassifier
from Defend_MI.BiDO import engine
from utils.bido_metrics import calculate_hsic

def apply_bido_defense(config, model:AbstractClassifier):

    class BiDOClassifier(model.__class__):

        def __init__(self, config):
            super(BiDOClassifier, self).__init__(config)
            self.only_logits = False
            if not config.dataset.val_drop_last: # drop_last must be True in the validation set dataloader because of how BiDO's test_HSIC function is implemented, this is possible to fix but hasn't been done yet
                raise ValueError("dataset.val_drop_last=False is not currently supported for BiDO defense. Please set 'dataset.val_drop_last=True' on the command line.")

        def train_one_epoch(self, train_loader): # adapted from DefendMI/BiDO/train_HSIC.py
            train_loss, train_perc_accuracy = engine.train_HSIC(self,
                                                      self.criterion.cuda(),
                                                      self.optimizer,
                                                      train_loader,
                                                      self.config.defense.a1,
                                                      self.config.defense.a2,
                                                      self.config.dataset.n_classes,
                                                      ktype=self.config.defense.ktype,
                                                      hsic_training=self.config.defense.hsic_training,
                                                      )
            return train_loss
        
        def evaluate(self, loader, train_set=False):
            self.to(self.device)

            loss, perc_accuracy = engine.test_HSIC(self,
                                              self.criterion.cuda(),
                                              loader,
                                              self.config.defense.a1,
                                              self.config.defense.a2,
                                              self.config.dataset.n_classes,
                                              ktype=self.config.defense.ktype,
                                              hsic_training=self.config.defense.hsic_training,
                                              )
            
            return loss, perc_accuracy/100 # convert accuracy from percentage to decimal

        def forward_without_intermediate_embeddings(self, x):
            z = self.feature_extractor(x)
            logits = self.classification_layer(z)
            return z, logits

        def forward_only_logits(self, x): # necessary for compatibility with attacks
            _, logits = self(x)
            return logits

        def forward(self, x): # adapted from DefendMI/BiDO/model.py
            embeddings_list = []
            arch = ""

            if type(self.feature_extractor) == nn.DataParallel:
                if type(self.feature_extractor.module) == torchvision.models.resnet.ResNet:
                    arch = 'resnet'
                elif self.config.model.architecture == "densenet169": #! TODO clean this up later
                    arch = 'densenet169'
                else:
                    arch = 'unknown' # no explicit intermediate layers have been determined for BiDO calculation, so will default to just using the output of the feature extractor
            elif type(self.feature_extractor) == torchvision.models.resnet.ResNet:
                arch = 'resnet'
            elif self.config.model.architecture == "densenet169": #! TODO clean this up later
                arch = 'densenet169'
            else:
                arch = 'unknown'

            if arch == 'resnet': # use intermediate outputs from each of the four major ResNet blocks
                x = self.feature_extractor.conv1(x)
                x = self.feature_extractor.bn1(x)
                x = self.feature_extractor.relu(x)
                x = self.feature_extractor.maxpool(x)

                hidden1 = self.feature_extractor.layer1(x)
                embeddings_list.append(hidden1)

                hidden2 = self.feature_extractor.layer2(hidden1)
                embeddings_list.append(hidden2)

                hidden3 = self.feature_extractor.layer3(hidden2)
                embeddings_list.append(hidden3)

                hidden4 = self.feature_extractor.layer4(hidden3)
                embeddings_list.append(hidden4)

                z = self.feature_extractor.avgpool(hidden4)
                z = torch.flatten(z, 1)
                logits = self.classification_layer(z)

            elif arch == 'densenet169':
                # assert nn.sequential as type of feature_extractor (#! not compatible with DataParallel yet)
                assert isinstance(self.feature_extractor, nn.Sequential), "Expected feature_extractor to be of type nn.Sequential for densenet169 architecture."
                for layer in self.feature_extractor:
                    x = layer(x)
                    if isinstance(layer, torchvision.models.densenet._DenseBlock):
                        embeddings_list.append(x) # save hiddens from four major DenseBlocks for use in BiDO calculation
                logits = self.classification_layer(x)
            else: # just use the final output before classification layer
                z = self.feature_extractor(x)
                embeddings_list.append(z)
                logits = self.classification_layer(z)

            if self.only_logits:
                return logits
            else:
                return embeddings_list, logits  


        def get_hiddens(self, x):
            hiddens, _ = self.forward(x)
            return hiddens

        def get_outputs(self, x):
            _, outputs = self.forward(x)
            return outputs

        
    bido_defended_model = BiDOClassifier(config)
    return bido_defended_model
        