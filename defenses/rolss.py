import torch.nn as nn

from RoLSS.models.torchvision.models import resnet, densenet # use the version of resnet provided by the RoLSS authors that is altered to include the skip connection

from classifiers.abstract_classifier import AbstractClassifier
from classifiers.pretrained import PreTrainedClassifier

def apply_RoLSS_defense(config, model:AbstractClassifier):

    if not isinstance(model, PreTrainedClassifier):
        raise ValueError("RoLSS is only currently supported for PreTrainedClassifiers. Supported architectures: resnet18, resnet34, resnet50, resnet101, resnet152")

    class RoLSS(model.__class__):
        
        def __init__(self, config):
            super(RoLSS, self).__init__(config)

        def init_model(self):

            # load the model again, this time with the skip parameter
            arch = config.model.architecture.lower()
            pretrained = config.model.pretrained

            if 'resnet' in arch:

                if arch == 'resnet18':
                    weights = resnet.ResNet18_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = resnet.resnet18(weights=weights, skip=config.defense.skip)
                elif arch == 'resnet34':
                    weights = resnet.ResNet34_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = resnet.resnet34(weights=weights, skip=config.defense.skip)
                elif arch == 'resnet50':
                    weights = resnet.ResNet50_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = resnet.resnet50(weights=weights, skip=config.defense.skip)
                elif arch == 'resnet101':
                    weights = resnet.ResNet101_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = resnet.resnet101(weights=weights, skip=config.defense.skip)
                elif arch == 'resnet152':    
                    weights = resnet.ResNet152_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = resnet.resnet152(weights=weights, skip=config.defense.skip)

                if self.config.dataset.n_classes != skip_defended_feature_extractor.fc.out_features:
                    # exchange the last layer to match the desired numbers of classes
                    skip_defended_feature_extractor.fc = nn.Linear(skip_defended_feature_extractor.fc.in_features, self.config.dataset.n_classes)

                classification_layer = skip_defended_feature_extractor.fc
                skip_defended_feature_extractor.fc = nn.Identity()

            #! densenet has a bug that is not yet fixed for RoLSS defense
            elif 'densenet' in arch:
                if arch == 'densenet121':
                    weights = densenet.DenseNet121_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = densenet.densenet121(weights=weights, skip=config.defense.skip)
                elif arch == 'densenet161':
                    weights = densenet.DenseNet161_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = densenet.densenet161(weights=weights, skip=config.defense.skip)
                elif arch == 'densenet169':
                    weights = densenet.DenseNet169_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = densenet.densenet169(weights=weights, skip=config.defense.skip)
                    # skip_defended_feature_extractor = densenet.densenet169(pretrained=pretrained, skip=config.defense.skip)
                elif arch == 'densenet201':
                    weights = densenet.DenseNet201_Weights.DEFAULT if pretrained else None
                    skip_defended_feature_extractor = densenet.densenet201(weights=weights, skip=config.defense.skip)
                
                if self.config.dataset.n_classes != skip_defended_feature_extractor.classifier.out_features:
                    # exchange the last layer to match the desired numbers of classes
                    skip_defended_feature_extractor.classifier = nn.Linear(skip_defended_feature_extractor.classifier.in_features, self.config.dataset.n_classes)
                
                classification_layer = skip_defended_feature_extractor.classifier
                skip_defended_feature_extractor = nn.Sequential(
                    *skip_defended_feature_extractor.features,
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )

            #! resnext not yet tested for RoLSS defense
            # elif 'resnext' in arch:
            #     if arch == 'resnext50':
            #         weights = resnet.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
            #         skip_defended_feature_extractor = resnet.resnext50_32x4d(weights=weights, skip=config.defense.skip)
            #     elif arch == 'resnext101':
            #         weights = resnet.ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None
            #         skip_defended_feature_extractor = resnet.resnext101_32x8d(weights=weights, skip=config.defense.skip)
            
            else:
                raise RuntimeError(
                    f'Model with the name {arch} not currently supported for RoLSS defense. Supported architectures: resnet18, resnet34, resnet50, resnet101, resnet152'
                )
        

            return skip_defended_feature_extractor, classification_layer

    RoLSS_defended_model = RoLSS(config)
    return RoLSS_defended_model