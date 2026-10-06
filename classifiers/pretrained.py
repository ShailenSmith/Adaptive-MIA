import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import densenet, inception, resnet, vgg

from classifiers.abstract_classifier import AbstractClassifier
from utils.torchutil import unwrapped_parallel_module

class PreTrainedClassifier(AbstractClassifier):
    
    def __init__(self, config):
        super(PreTrainedClassifier, self).__init__(config)
        
    def init_model(self): # adapted from _build_model() in Plug-and-Play-Attacks/models/classifier.py
        super(PreTrainedClassifier, self).init_model()
        self.n_channels = self.config.dataset.input_size[0]

        arch = self.config.model.architecture.lower().replace('-',
                                                             '').replace('_',
                                                                        '').strip()
        pretrained = self.config.model.pretrained
        if 'resnet' in arch:
            if arch == 'resnet18':
                weights = resnet.ResNet18_Weights.DEFAULT if pretrained else None
                model = resnet.resnet18(weights=weights)
            elif arch == 'resnet34':
                weights = resnet.ResNet34_Weights.DEFAULT if pretrained else None
                model = resnet.resnet34(weights=weights)
            elif arch == 'resnet50':
                weights = resnet.ResNet50_Weights.DEFAULT if pretrained else None
                model = resnet.resnet50(weights=weights)
            elif arch == 'resnet101':
                weights = resnet.ResNet101_Weights.DEFAULT if pretrained else None
                model = resnet.resnet101(weights=weights)
            elif arch == 'resnet152':
                weights = resnet.ResNet152_Weights.DEFAULT if pretrained else None
                model = resnet.resnet152(weights=weights)
            else:
                raise RuntimeError(
                    f'No ResNet with the name {arch} available'
                )

            if self.config.dataset.n_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.config.dataset.n_classes)

            classification_layer = model.fc
            model.fc = nn.Identity()
            
        #  # VGG support #! TODO: Not tested yet 
        # elif 'vgg' in arch:
            
        #     if arch == 'vgg11':
        #         weights = vgg.VGG11_Weights.DEFAULT if pretrained else None
        #         model = vgg.vgg11(weights=weights)
        #     elif arch == 'vgg13':
        #         weights = vgg.VGG13_Weights.DEFAULT if pretrained else None
        #         model = vgg.vgg13(weights=weights)
        #     elif arch == 'vgg16':
        #         weights = vgg.VGG16_Weights.DEFAULT if pretrained else None
        #         model = vgg.vgg16(weights=weights)
        #     elif arch == 'vgg19':
        #         weights = vgg.VGG19_Weights.DEFAULT if pretrained else None
        #         model = vgg.vgg19(weights=weights)
        #     else:
        #         raise RuntimeError(
        #             f'No VGG with the name {arch} available'
        #         )

        #     if self.config.dataset.n_classes != model.classifier[-1].out_features:
        #         # exchange the last layer to match the desired numbers of classes
        #         in_features = model.classifier[-1].in_features
        #         model.classifier[-1] = nn.Linear(in_features, self.config.dataset.n_classes)

        #     classification_layer = model.classifier
            
        #     model = model.features  # Extract the feature extractor part
        #     model.classifier = nn.Identity()

        elif 'densenet' in arch:
            if arch == 'densenet121':
                weights = densenet.DenseNet121_Weights.DEFAULT if pretrained else None
                model = densenet.densenet121(weights=weights)
            elif arch == 'densenet161':
                weights = densenet.DenseNet161_Weights.DEFAULT if pretrained else None
                model = densenet.densenet161(weights=weights)
            elif arch == 'densenet169':
                weights = densenet.DenseNet169_Weights.DEFAULT if pretrained else None
                model = densenet.densenet169(weights=weights)
            elif arch == 'densenet201':
                weights = densenet.DenseNet201_Weights.DEFAULT if pretrained else None
                model = densenet.densenet201(weights=weights)
            else:
                raise RuntimeError(
                    f'No DenseNet with the name {arch} available'
                )

            if self.config.dataset.n_classes != model.classifier.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.classifier = nn.Linear(model.classifier.in_features, self.config.dataset.n_classes)
            
                    
            classification_layer = model.classifier
            # model = model.features  # Extract the feature extractor part
            
            model = nn.Sequential( #! BiDO implementation for densenet depends on this Sequential wrapper at the moment
                *model.features,
                nn.ReLU(inplace=True), # from the original DenseNet forward pass
                nn.AdaptiveAvgPool2d((1, 1)), # from the original DenseNet forward pass
                nn.Flatten(), # from the original DenseNet forward pass
            )
            
            # model.classifier = nn.Identity()
        
        elif 'resnest' in arch:
            torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
            if arch == 'resnest50':
                model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=pretrained)
            elif arch == 'resnest101':
                model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=pretrained)
            elif arch == 'resnest200':
                model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest200', pretrained=pretrained)
            elif arch == 'resnest269':
                model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest269', pretrained=pretrained)
            else:
                raise RuntimeError(
                    f'No ResNeSt with the name {arch} available'
                )
            
            if self.config.dataset.n_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.config.dataset.n_classes)

            classification_layer = model.fc
            model.fc = nn.Identity()

        elif 'resnext' in arch:
            if arch == 'resnext50':
                weights = resnet.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
                model = resnet.resnext50_32x4d(weights=weights)
            elif arch == 'resnext101':
                weights = resnet.ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None
                model = resnet.resnext101_32x8d(weights=weights)
            else:
                raise RuntimeError(
                    f'No ResNeXt with the name {arch} available'
                )
            
            if self.config.dataset.n_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.config.dataset.n_classes)

            classification_layer = model.fc
            model.fc = nn.Identity()

        elif 'inception' in arch:
            weights = inception.Inception_V3_Weights.DEFAULT if pretrained else None
            model = inception.inception_v3(weights=weights,
                                           aux_logits=True,
                                           init_weights=False)   

            if self.config.dataset.n_classes != model.fc.out_features:
                # exchange the last layer to match the desired numbers of classes
                model.fc = nn.Linear(model.fc.in_features, self.config.dataset.n_classes)
            
            classification_layer = model.fc
            model.fc = nn.Identity()
            
        else:
            raise RuntimeError(
                f'No model with the name {arch} available'
            )

        return model, classification_layer
    
    def forward(self, x):
        
        if isinstance(unwrapped_parallel_module(self.feature_extractor), inception.Inception3):
            if self.training:
                z, aux_logits = self.feature_extractor(x)
                logits = self.classification_layer(z)
                return logits, aux_logits
            else:
                z = self.feature_extractor(x)
                logits = self.classification_layer(z)
                return logits
        
        else:
            z = self.feature_extractor(x)
            logits = self.classification_layer(z)
            return logits

    
    # Only used when training inception models
    def get_inception_loss(self, inception_output, target):
        
        if self.training:
            logits, aux_logits = inception_output
            aux_loss = torch.tensor(0.0, device=self.device)
            aux_loss += self.criterionSum(aux_logits, target) #! does this need to be criterionSum instead of criterion too?
            loss = self.criterionSum(logits, target) + aux_loss
        else:
            logits = inception_output
            loss = self.criterionSum(logits, target)
        
        return loss
    
    def get_loss(self, output, target): # overwritten from AbstractClassifier

        if isinstance(unwrapped_parallel_module(self.feature_extractor), inception.Inception3):
            loss = self.get_inception_loss(output, target)

        else:
            loss = self.criterionSum(output, target)

        return loss
