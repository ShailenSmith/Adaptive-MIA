
#! WIP -- don't push to public yet
import os

import numpy as np
import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import TensorDataset, Dataset
from torchvision.transforms.transforms import Resize
from tqdm import tqdm

from Plug_and_Play_Attacks.datasets.custom_subset import SingleClassSubset

# Only used when cpu_optimize=True. A small worker count is enough to keep image
# decode/transform overlapped with the GPU forward pass without oversubscribing CPUs
# that GPUHelper may be sharing across concurrent runs.
CPU_OPTIMIZE_NUM_WORKERS = min(4, os.cpu_count() or 1)


def _get_indices_by_class(dataset):
    """Group `dataset`'s sample indices by class, computed once and cached on the
    dataset object itself -- so if the same dataset instance is reused across several
    DistanceEvaluation calls (e.g. cpu_optimize's shared training-dataset cache in
    evaluate_attack), the O(N) targets->array conversion that SingleClassSubset would
    otherwise redo on every single-class lookup only happens once total, not once per
    attacked class per caller."""
    cached = getattr(dataset, '_cpu_optimize_indices_by_class', None)
    if cached is not None:
        return cached
    targets_array = np.array(dataset.targets)
    indices_by_class = {int(cls): np.where(targets_array == cls)[0]
                        for cls in np.unique(targets_array)}
    dataset._cpu_optimize_indices_by_class = indices_by_class
    return indices_by_class


class _CachedClassSubset(Dataset):
    """Behavior-identical to SingleClassSubset, but takes a precomputed
    indices-by-class map instead of rebuilding it from `dataset.targets` every time."""

    def __init__(self, dataset, target_class, indices_by_class):
        self.dataset = dataset
        self.indices = indices_by_class[target_class]
        self.target_class = target_class

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class DistanceEvaluation():

    def __init__(self, model, img_size: int, train_set: Dataset):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.img_size = img_size
        self.train_set = train_set

    def compute_dist(self, imgs, targets, batch_size=64, resize=None, rtpt=None,
                      cpu_optimize=False):
        """Compute average minimum squared L2 distance between attack and training embeddings.

        Args:
            imgs: Tensor [N, C, H, W] or a Dataset whose order matches `targets`
            targets: Tensor [N] of class IDs
            resize: if set, resize attack images to (resize, resize) before embedding
            cpu_optimize: if True, avoid CPU-bound overhead that doesn't affect the
                result: skip SingleClassSubset's repeated O(N) targets->array
                conversion (cached per dataset instead), and load images with worker
                processes + pinned memory instead of serially on the main process.
                Default False reproduces the exact prior behavior.
        """
        self.model.eval()
        self.model.to(self.device)
        resize_tf = T.Resize((resize, resize), antialias=True) if resize else None

        loader_kwargs = {}
        if cpu_optimize:
            loader_kwargs['num_workers'] = CPU_OPTIMIZE_NUM_WORKERS
            loader_kwargs['pin_memory'] = self.device == 'cuda'

        indices_by_class = _get_indices_by_class(self.train_set) if cpu_optimize else None

        target_values = list(dict.fromkeys(targets.cpu().tolist()))  # preserve order
        smallest_distances = []
        mean_distances_list = [['target', 'mean_dist']]

        for step, target in tqdm(enumerate(target_values), desc='Feature distance', total=len(target_values)):
            mask = targets == target

            # Build attack loader for this class
            if isinstance(imgs, torch.Tensor):
                imgs_masked = imgs[mask]
                attack_loader = DataLoader(
                    TensorDataset(imgs_masked), batch_size, shuffle=False, **loader_kwargs)
                def _get_img(batch):
                    return batch[0]
            else:
                indices = mask.nonzero(as_tuple=False).squeeze(1).tolist()
                attack_loader = DataLoader(
                    Subset(imgs, indices), batch_size, shuffle=False, **loader_kwargs)
                def _get_img(batch):
                    return batch[0]  # (img, label) from underlying dataset

            # Embed training images for this class
            if cpu_optimize:
                target_subset = _CachedClassSubset(self.train_set, target, indices_by_class)
            else:
                target_subset = SingleClassSubset(self.train_set, target_class=target)
            target_embeddings = []
            for x, _ in DataLoader(target_subset, batch_size, **loader_kwargs):
                with torch.no_grad():
                    target_embeddings.append(self.model(x.to(self.device)).cpu())
            target_embeddings = torch.cat(target_embeddings, dim=0)

            # Embed attack images for this class
            attack_embeddings = []
            for batch in attack_loader:
                with torch.no_grad():
                    img_batch = _get_img(batch).to(self.device)
                    if resize_tf is not None:
                        img_batch = resize_tf(img_batch)
                    attack_embeddings.append(self.model(img_batch).cpu())
            attack_embeddings = torch.cat(attack_embeddings, dim=0)

            distances = torch.cdist(attack_embeddings, target_embeddings, p=2).cpu()
            distances = distances ** 2
            distances, _ = torch.min(distances, dim=1)
            smallest_distances.append(distances.cpu())
            mean_distances_list.append([target, distances.cpu().mean().item()])

            if rtpt:
                rtpt.step(
                    subtitle=f'Distance Evaluation step {step} of {len(target_values)}')

        smallest_distances = torch.cat(smallest_distances, dim=0)
        return smallest_distances.mean(), mean_distances_list

    def find_closest_training_sample(self, imgs, targets, batch_size=64):
        self.model.eval()
        self.model.to(self.device)
        closest_imgs = []
        smallest_distances = []
        resize = Resize((self.img_size, self.img_size), antialias=True)
        for img, target in zip(imgs, targets):
            img = img.to(self.device)
            img = resize(img)
            if torch.is_tensor(target):
                target = target.cpu().item()
            target_subset = SingleClassSubset(self.train_set, target_class=target)
            if len(img) == 3:
                img = img.unsqueeze(0)
            target_embeddings = []
            with torch.no_grad():
                output_img = self.model(img).cpu()
                for x, _ in DataLoader(target_subset, batch_size):
                    target_embeddings.append(self.model(x.to(self.device)).cpu())
            target_embeddings = torch.cat(target_embeddings, dim=0)
            distances = torch.cdist(output_img, target_embeddings, p=2)
            distances = distances ** 2
            distance, idx = torch.min(distances, dim=1)
            smallest_distances.append(distance.item())
            closest_imgs.append(target_subset[idx.item()][0])
        return closest_imgs, smallest_distances
