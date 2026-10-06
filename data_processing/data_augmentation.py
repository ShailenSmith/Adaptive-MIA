
from torchvision import datasets, transforms
from torchvision.transforms import v2 as transforms_v2

import omegaconf


def get_transforms(config, extra_augmentations:list = [], train = True): #! TODO: Make it give training or test augmentations
    # Get dataaugmentations
    image_height = config.dataset.input_size[1]
    image_width  = config.dataset.input_size[2]
    image_size = (image_height, image_width)
    augmentations = []

    if config.dataset.resize:
        resize = transforms.Resize(image_size, antialias=True)
        augmentations.append(resize)

    if config.dataset.augment_data and train:

        use_random_resized_crop = getattr(config.dataset.transformations, "use_random_resized_crop", True)
        use_color_jitter = getattr(config.dataset.transformations, "use_color_jitter", True)
        use_random_horizontal_flip = getattr(config.dataset.transformations, "use_random_horizontal_flip", True)
        use_gaussian_blur = getattr(config.dataset.transformations, "use_gaussian_blur", False)

        if use_random_resized_crop:
            random_resize_crop = transforms.RandomResizedCrop(**config.dataset.transformations.RandomResizedCrop)
            augmentations.append(random_resize_crop)
        if use_random_horizontal_flip:
            random_horizontal_flip = transforms.RandomHorizontalFlip(**config.dataset.transformations.RandomHorizontalFlip)
            augmentations.append(random_horizontal_flip)
        if use_color_jitter:
            if config.dataset.input_size[0] == 3:
                color_jitter = transforms.ColorJitter(**config.dataset.transformations.ColorJitter)
                augmentations.append(color_jitter)
        if use_gaussian_blur:
            gaussian_blur_cfg = getattr(config.dataset.transformations, "GaussianBlur", {})
            gaussian_blur = transforms.GaussianBlur(**gaussian_blur_cfg)
            augmentations.append(gaussian_blur)
    
    if config.dataset.augment_data and not train:
        center_crop = transforms.CenterCrop((image_height, image_width)) #! Already resized?
        augmentations += [center_crop]
        
    if extra_augmentations:
        augmentations += extra_augmentations

    try: # ensure backcompability with older configs
        normalize = config.dataset.normalize
    except omegaconf.errors.ConfigAttributeError:
        normalize = True
        print("Warning: dataset.normalize not specified in config. Setting it to True by default.")
            
    if config.dataset.dataset == "CIFAR10":
        if normalize:
            base_transformations = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        else:
            base_transformations = [
                transforms.ToTensor(),
            ]
        # base_transformations = [
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ]
        
        all_transformations = augmentations + base_transformations
        transform = transforms.Compose(all_transformations)
        
    elif config.dataset.dataset == "MNIST":
        if normalize:
            base_transformations = [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        else:
            base_transformations = [
                transforms.ToTensor(),
            ]
        # base_transformations = [
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ]
        
        all_transformations = augmentations + base_transformations        
        transform = transforms.Compose(all_transformations)
        
        
    elif config.dataset.dataset == "FashionMNIST":
        if normalize:
            base_transformations = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ]
        else:
            base_transformations = [
                transforms.ToTensor(),
                ]
        # base_transformations = [
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5,), (0.5,))
        # ]
        
        all_transformations = augmentations + base_transformations
        
        transform = transforms.Compose(all_transformations)
        
    elif config.dataset.dataset == "CelebA":
        if normalize:
            base_transformations = ([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            base_transformations = ([
                transforms.ToTensor()])
    
        all_transformations = augmentations + base_transformations
    
        transform = transforms.Compose(all_transformations)
    
    elif config.dataset.dataset == "FaceScrub":
        if normalize:
            base_transformations = ([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            base_transformations = ([
                transforms.ToTensor()])
        
        all_transformations = augmentations + base_transformations
        
        transform = transforms.Compose(all_transformations)
        
        
    elif config.dataset.dataset == "stanford_dogs":
        if normalize:
            base_transformations = ([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            base_transformations = ([
                transforms.ToTensor()])

        # base_transformations = ([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        all_transformations = augmentations + base_transformations
        
        transform = transforms.Compose(all_transformations)
    
    else:
        raise ValueError(f"Unknown dataset: {config.dataset.dataset}")

    return transform


def get_batch_transforms(config):
    """Return batch-level transforms (CutMix, MixUp) for use in the training loop.

    These operate on (images_batch, labels_batch) tuples and cannot be used
    inside transforms.Compose. Apply them per-batch, e.g.:

        batch_transform = get_batch_transforms(config)
        for images, labels in dataloader:
            images, labels = batch_transform(images, labels)

    Returns None if neither CutMix nor MixUp is enabled.
    """
    use_cutmix = getattr(config.dataset.transformations, "use_cutmix", False)
    use_mixup = getattr(config.dataset.transformations, "use_mixup", False)

    candidates = []
    if use_cutmix:
        cutmix_cfg = getattr(config.dataset.transformations, "CutMix", {})
        candidates.append(transforms_v2.CutMix(alpha=cutmix_cfg.get("alpha", 1.0),
                                               num_classes=config.dataset.n_classes))
    if use_mixup:
        mixup_cfg = getattr(config.dataset.transformations, "MixUp", {})
        candidates.append(transforms_v2.MixUp(alpha=mixup_cfg.get("alpha", 1.0),
                                             num_classes=config.dataset.n_classes))

    if not candidates:
        return None
    
    #! defense compatibility check, handle more gracefully at some point
    if candidates and config.defense.name not in ["no_defense", "label_smoothing"]:
        raise NotImplementedError("Batch-level augmentations (CutMix, MixUp) are not currently compatible with defenses besides LabelSmoothing. Please set use_cutmix and use_mixup to False when using a defense.")

    # Randomly apply one of CutMix or MixUp each batch
    return transforms_v2.RandomChoice(candidates)