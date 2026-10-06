from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split
import torch
import os

import data_processing.celeba as celeba
import data_processing.facescrub as facescrub
import data_processing.stanford_dogs as stanford_dogs
import omegaconf

def get_datasets(config, train_transform, test_transform):
    """Dynamically loads datasets based on the configuration."""
    
    if config.dataset.dataset == "CIFAR10":
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        full_val_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        
    elif config.dataset.dataset == "MNIST":   
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        full_val_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=test_transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        
    elif config.dataset.dataset == "FashionMNIST":
        full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
        full_val_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=test_transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)
        
    elif config.dataset.dataset == "FaceScrub":
        # Only runs if it is not downloaded
        facescrub.download() 
        facescrub.process_zip()
        
        full_train_dataset = facescrub.FaceScrub(group = config.dataset.group,
                                                 train = True,
                                                 split_seed = config.training.seed,
                                                 transform = train_transform)
        
        full_val_dataset = facescrub.FaceScrub(group = config.dataset.group,
                                               train = True,
                                               split_seed = config.training.seed,
                                               transform = test_transform)
        
        test_dataset = facescrub.FaceScrub(group = config.dataset.group,
                                            train = False,
                                            split_seed = config.training.seed,
                                            transform = test_transform)
        
    elif config.dataset.dataset == "stanford_dogs":
        # Only runs if it is not downloaded
        stanford_dogs.download()
        stanford_dogs.process()
        
        full_train_dataset = stanford_dogs.StanfordDogs(train=True,
                                                        cropped=config.dataset.cropped,
                                                        split_seed=config.training.seed,
                                                        transform=train_transform,
                                                        root="data/stanford_dogs",
                                                        )
        
        full_val_dataset = stanford_dogs.StanfordDogs(train=True,
                                                        cropped=config.dataset.cropped,
                                                        split_seed=config.training.seed,
                                                        transform=test_transform,
                                                        root="data/stanford_dogs",
                                                        )
    
        test_dataset = stanford_dogs.StanfordDogs(train=False,
                                                    cropped=config.dataset.cropped,
                                                    split_seed=config.training.seed,
                                                    transform=test_transform,
                                                    root="data/stanford_dogs",
                                                    )
        
    elif config.dataset.dataset == "CelebA": 
        # Backwards compatibility for old configs
        try:
            use_hdcrop = config.dataset.use_hdcrop
        except omegaconf.errors.ConfigAttributeError:
            use_hdcrop = False
            print("Warning: 'use_hd_crop' not found in config. Defaulting to False.")
            
        necessary_files = ["data/celeba/list_eval_partition.txt", 
                           "data/celeba/identity_CelebA.txt"
                           "data/celeba/list_attr_celeba.txt",
                           "data/celeba/list_bbox_celeba.txt",
                           "data/celeba/list_landmarks_align_celeba.txt"]
        
        if use_hdcrop:
            necessary_files.append("data/celeba/hdcrop")
            img_folder = "hdcrop"
        else:
            necessary_files.append("data/celeba/img_align_celeba")
            img_folder = "img_align_celeba"
        
        is_downloaded = all(map(os.path.exists, necessary_files))
        
        # Check if data is downloaded
        if not is_downloaded:
            celeba.download()
            celeba.process(file_name=img_folder + ".zip")
            
        
        full_train_dataset = celeba.CelebA_N_most_common(train=True,
                                   split_seed=config.training.seed,
                                   transform=train_transform,
                                   root='./data/celeba',
                                   N = config.dataset.n_classes,
                                   img_folder=img_folder
                                   )
        
        full_val_dataset = celeba.CelebA_N_most_common(train=True,
                                    split_seed=config.training.seed,
                                    transform=test_transform,
                                    root='./data/celeba',
                                    N = config.dataset.n_classes,
                                    img_folder=img_folder
                                    )
        
        test_dataset = celeba.CelebA_N_most_common(train=False,
                                   split_seed=config.training.seed,
                                   transform=test_transform,
                                   root='./data/celeba',
                                   N = config.dataset.n_classes,
                                   img_folder=img_folder
                                   )

    else:
        raise ValueError(f"Unknown dataset: {config.dataset.dataset}")

    # Split the full training dataset into training and validation sets
    generator1 = torch.Generator().manual_seed(config.training.seed)
    train_size = int((1 - config.training.validation_size) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # code from https://discuss.pytorch.org/t/changing-transforms-after-creating-a-dataset/64929/8
    indices = torch.randperm(len(full_train_dataset), generator=generator1)
    train_dataset = torch.utils.data.Subset(full_train_dataset, indices[:-val_size])
    val_dataset = torch.utils.data.Subset(full_val_dataset, indices[-val_size:])
    # train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator1)
    
    train_dataset = AttackDataset(train_dataset)

    #! Maybe also do: test_dataset = AttackDataset(test_dataset)
    
    try:
        if config.dataset.permute_train_labels:
            generator_perm = torch.Generator().manual_seed(config.training.seed)
            shuffled_indices = torch.randperm(len(train_dataset.targets), generator=generator_perm)
            train_dataset.targets = train_dataset.targets[shuffled_indices]
    except omegaconf.errors.ConfigAttributeError as e:
        print(f"WARNING: {e}")

    return train_dataset, val_dataset, test_dataset


def load_example_images(dataset_name, class_label, num_images=1, seed=42):
    """Load up to `num_images` real example images of `class_label` from `dataset_name`
    ('FaceScrub', 'CelebA', or 'stanford_dogs'), using this project's standard default
    dataset parameters (see configuration/classifier/dataset/*.yaml) -- for use when only
    the bare dataset name is known, e.g. reconstructing reference images from a saved
    run's 'dataset' config field, without the full training config.

    Returns a list of up to `num_images` FloatTensor [3, H, W] images in [0, 1] (fewer if
    the class has fewer examples than that), or [] if the dataset/class isn't found.
    """
    transform = transforms.ToTensor()

    if dataset_name == "FaceScrub":
        facescrub.download()
        facescrub.process_zip()
        dataset = facescrub.FaceScrub(group="all", train=True, split_seed=seed, transform=transform)

    elif dataset_name == "CelebA":
        img_folder = "hdcrop"
        necessary_files = ["data/celeba/list_eval_partition.txt",
                           "data/celeba/identity_CelebA.txt",
                           "data/celeba/list_attr_celeba.txt",
                           "data/celeba/list_bbox_celeba.txt",
                           "data/celeba/list_landmarks_align_celeba.txt",
                           f"data/celeba/{img_folder}"]
        if not all(map(os.path.exists, necessary_files)):
            celeba.download()
            celeba.process(file_name=img_folder + ".zip")
        dataset = celeba.CelebA_N_most_common(train=True, split_seed=seed, transform=transform,
                                              root='./data/celeba', N=1000, img_folder=img_folder)

    elif dataset_name == "stanford_dogs":
        stanford_dogs.download()
        stanford_dogs.process()
        dataset = stanford_dogs.StanfordDogs(train=True, cropped=True, split_seed=seed,
                                             transform=transform, root="data/stanford_dogs")

    else:
        print(f"WARNING: Unknown dataset '{dataset_name}', cannot load example image.")
        return []

    if class_label not in dataset.targets:
        return []
    indices = [i for i, t in enumerate(dataset.targets) if t == class_label][:num_images]
    return [dataset[i][0] for i in indices]


class AttackDataset(Dataset):
    """
    Wrap a subset to add a 'self.targets' attribute. Needed for attacks

    Args:
        subset (torch.utils.data.Subset): A subset of a dataset.
    """
    
    def __init__(self, subset):
        self.subset = subset
        self.targets = torch.tensor([subset.dataset.targets[idx] for idx in subset.indices])

    def __getitem__(self, index):
        im, _ = self.subset[index]
        return im, self.targets[index]

    def __len__(self):
        return len(self.subset)

