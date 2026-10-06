from collections import OrderedDict
import torch
import torch.nn as nn

from hcrbounds.codes import testers
from classifiers.abstract_classifier import AbstractClassifier
from classifiers.pretrained import PreTrainedClassifier

def calculate_hcr(model: AbstractClassifier, loader, hcr_config, hcr_run_name="hcr_test"): # calculate Hammersley-Chapman-Robbins bounds using the hcrbounds repo

    if isinstance(model, PreTrainedClassifier) and model.config.model.architecture == "resnet18":
        model_name = 'ResNet18'

        if hcr_config.include_avgpool: # include avgpool layer in hcr_model to decrease the size of the feature embeddings used to calculate the HCR bounds
            hcr_model = nn.Sequential(OrderedDict([
                *(list(model.feature_extractor.named_children())[:-1]),
                ('flatten', torch.nn.Flatten(1)),
            ]))
            hcr_final = nn.Sequential(OrderedDict([
                *(list(model.feature_extractor.named_children())[-1:]),
                ('clfn_layer', model.classification_layer)
            ]))
        else:
            hcr_model = nn.Sequential(OrderedDict([
                *(list(model.feature_extractor.named_children())[:-2]),
            ]))
            hcr_final = nn.Sequential(OrderedDict([
                *(list(model.feature_extractor.named_children())[-2:]),
                ('flatten', torch.nn.Flatten(1)),
                ('clfn_layer', model.classification_layer)
            ]))
  
    else:
        raise RuntimeError("As of now, only ResNet18 is supported for HCR bound calculation")

    if hcr_config.use_custom_indiff: # define the custom indiff here
        custom_indiff = None
    else:
        custom_indiff = None

    hcr, hcrmax = testers.hcr(loader, hcr_model, hcr_final,
                    num_batches=hcr_config.num_batches,
                    seed_torch = model.config.training.seed,
                    name=model_name,
                    num_iter=hcr_config.num_iter,
                    num_pits=hcr_config.num_pits,
                    hcr_run_name=hcr_run_name,
                    custom_indiff=custom_indiff,
                    sigma_scale=hcr_config.sigma_scale,
                    )

    return hcr, hcrmax