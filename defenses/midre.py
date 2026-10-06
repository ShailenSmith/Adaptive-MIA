from torchvision import transforms

from classifiers.abstract_classifier import AbstractClassifier


def apply_midre_defense(config, model: AbstractClassifier):
    class MIDREDefense(model.__class__):
        def train_model(self, train_loader, val_loader):
            dataset = train_loader.dataset.subset.dataset
            augmentations = list(dataset.transform.transforms)
            random_erasing = transforms.RandomErasing(
                p=config.defense.RandomErasing.p,
                scale=tuple(config.defense.RandomErasing.scale),
                ratio=tuple(config.defense.RandomErasing.ratio),
                value=tuple(config.defense.RandomErasing.value),
            )

            for index, transform in enumerate(augmentations):
                if isinstance(transform, transforms.Normalize):
                    augmentations.insert(index + 1, random_erasing)
                    break
            else:
                raise ValueError("MIDRE requires a Normalize transform before random erasing.")

            dataset.transform = transforms.Compose(augmentations)
            return super().train_model(train_loader, val_loader)

    return MIDREDefense(config)
