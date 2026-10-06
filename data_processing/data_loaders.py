from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split
from data_processing.datasets import get_datasets
from data_processing.data_augmentation import get_transforms

def get_data_loaders(config):
    """Dynamically loads datasets based on the configuration."""
    
    train_transform = get_transforms(config, train=True)
    test_transform = get_transforms(config, train=False)
    
    train_dataset, val_dataset, test_dataset = get_datasets(config = config, 
                                                            train_transform=train_transform,
                                                            test_transform=test_transform)

    num_workers = config.training.dataloader_num_workers
    if 'val_drop_last' not in config.dataset.keys():
        val_drop_last = False
    else:
        val_drop_last = config.dataset.val_drop_last

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=config.model.hyper.batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            drop_last=True,
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=config.model.hyper.batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            drop_last=val_drop_last,)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=config.model.hyper.batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            drop_last=val_drop_last,)
    
    print(f"Data loaders for {config.dataset.dataset} loaded successfully")
    
    return train_loader, val_loader, test_loader