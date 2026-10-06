import torchvision
import torch
import torch.nn as nn

from tqdm import tqdm
import pickle
import os

from classifiers.abstract_classifier import AbstractClassifier

from Defend_MI.BiDO import utils, hsic
from RoLSS import models as RoLSS_models

def calculate_hsic(model:AbstractClassifier, loader, hsic_config=None): # adapted from multilayer_hsic function in Defend_MI/BiDO/engine.py
    # set n_classes and yz_ktype
    n_classes = model.classification_layer.out_features
    if hsic_config:
        yz_ktype = hsic_config.yz_ktype # should match whatever we set in the config
    else:
        yz_ktype = 'linear'
    
    # add model to device, get number of hidden outputs
    model.to(model.device)
    with torch.no_grad():
        n_hidden_outputs = len(get_hiddens(model, next(iter(loader))[0].to(model.device))) # get number of outputs from get_hiddens function

    # get hsic lists ready
    hsic_xz_lists = [] # (n_hidden_outputs, len(loader))
    hsic_yz_lists = [] # (n_hidden_outputs, len(loader))
    for i in range(n_hidden_outputs):
        hsic_xz_lists.append([])
        hsic_yz_lists.append([])

    # main loop to calculate HSIC
    with torch.no_grad():
        for batch_idx, (inputs, iden) in tqdm(enumerate(loader), 
                                            total=len(loader),
                                            desc="Calculating HSIC",
                                            ):
            inputs, iden = inputs.to(model.device), iden.to(model.device)
            iden = iden.view(-1)
            bs = inputs.size(0)

            hiddens = get_hiddens(model, inputs)
            h_target = utils.to_categorical(iden, num_classes=n_classes).float()
            h_data = inputs.view(bs, -1)

            if not isinstance(hiddens, list):
                hiddens = [hiddens]

            for i, hidden in enumerate(hiddens):
                hidden = hidden.view(bs, -1)

                
                hsic_xz, hsic_yz = hsic.hsic_objective(
                    hidden,
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=5.,
                    ktype=yz_ktype,
                )

                hsic_xz_lists[i].append(round(hsic_xz.item(), 5))
                hsic_yz_lists[i].append(round(hsic_yz.item(), 5))

    return hsic_xz_lists, hsic_yz_lists


def get_hiddens(model, x):
    if hasattr(model, 'get_hiddens'):
        return model.get_hiddens(x)
    else:
        embeddings_list = []
        arch = ""

        if type(model.feature_extractor) == nn.DataParallel:
            if type(model.feature_extractor.module) == torchvision.models.resnet.ResNet:
                arch = 'resnet'
            elif type(model.feature_extractor.module) == RoLSS_models.torchvision.models.resnet.ResNet: # allow compatibility with RoLSS models, which change the architecture slightly
                arch = 'resnet'
            else:
                arch = 'unknown' # no explicit intermediate layers have been determined for BiDO calculation, so will default to just using the output of the feature extractor
        else:
            if type(model.feature_extractor) == torchvision.models.resnet.ResNet:
                arch = 'resnet'
            elif type(model.feature_extractor) == RoLSS_models.torchvision.models.resnet.ResNet: # allow compatibility with RoLSS models, which change the architecture slightly
                arch = 'resnet'
            else:
                arch = 'unknown'

        if arch == 'resnet': # use intermediate outputs from each of the four major ResNet blocks
            x = model.feature_extractor.conv1(x)
            x = model.feature_extractor.bn1(x)
            x = model.feature_extractor.relu(x)
            x = model.feature_extractor.maxpool(x)

            hidden1 = model.feature_extractor.layer1(x)
            embeddings_list.append(hidden1)

            hidden2 = model.feature_extractor.layer2(hidden1)
            embeddings_list.append(hidden2)

            hidden3 = model.feature_extractor.layer3(hidden2)
            embeddings_list.append(hidden3)

            hidden4 = model.feature_extractor.layer4(hidden3)
            embeddings_list.append(hidden4)

            # z = model.feature_extractor.avgpool(hidden4)
            # z = torch.flatten(z, 1)
            # logits = model.classification_layer(z)
        
        else: # just use the final output before classification layer
            print("Warning: only ResNet models have been tested for HSIC calculation. Using the final output before classification layer to measure HSIC.")
            z = model.feature_extractor(x)
            embeddings_list.append(z)
            logits = model.classification_layer(z)

        return embeddings_list
