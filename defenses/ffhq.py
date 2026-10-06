from torch.utils.data import ConcatDataset, DataLoader
import torch
from torchvision import transforms

from classifiers.abstract_classifier import AbstractClassifier
from data_processing.ffhq import FFHQDataset

def apply_ffhq_defense(config, model:AbstractClassifier):
    
    class FFHQDefense(model.__class__):
        
        def __init__(self, config):
            config.dataset.n_classes = config.dataset.n_classes + 1
            
            super(FFHQDefense, self).__init__(config)
        
        
        def train_model(self, train_loader, val_loader):
            
            dataset = train_loader.dataset
            
            base_transformations = ([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
            base_transformations = transforms.Compose(base_transformations)
            
            ffhq_dataset = FFHQDataset(
                root=self.config.defense.path,
                label=self.config.dataset.n_classes - 1, # the label for FFHQ (zero-indexed and has already been incremented)
                transform=base_transformations 
            )
            
            new_train_dataset = ConcatDataset([dataset, ffhq_dataset])
            
            new_train_loader = DataLoader(
                new_train_dataset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory,
                drop_last=train_loader.drop_last
            )
            
            return super().train_model(new_train_loader, val_loader)
        
        
    ffhq_defended_model = FFHQDefense(config)
    return ffhq_defended_model
