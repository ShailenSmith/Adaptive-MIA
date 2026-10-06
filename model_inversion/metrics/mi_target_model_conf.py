
#! WIP -- don't push to public yet
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Only used when cpu_optimize=True -- see model_inversion.metrics.mi_distance_metrics.
CPU_OPTIMIZE_NUM_WORKERS = min(4, os.cpu_count() or 1)


class TargetModelConfidence():

    def __init__(self, target_model, device='cuda:0'):
        self.target_model = target_model
        self.device = device

    def compute_conf(self, dataset, targets, batch_size=64, cpu_optimize=False):
        """Compute target model confidence on reconstructed images.

        Confidence is the softmax probability assigned to the true target class,
        distinct from evaluation-model confidence (total_avg_conf / correct_avg_conf).

        Args:
            dataset: Dataset of attack images whose order matches `targets`
            targets: Tensor [N] of original class IDs
            batch_size: DataLoader batch size
            cpu_optimize: if True, load images with worker processes + pinned memory
                instead of serially on the main process (doesn't affect the result).
                Default False reproduces the exact prior behavior.

        Returns:
            avg_conf: float, mean target-class confidence across all images
            conf_by_class_list: list[list], CSV rows [['target', 'target_model_avg_conf'], ...]
        """
        self.target_model.eval()
        self.target_model.to(self.device)

        all_confs = []
        loader_kwargs = {}
        if cpu_optimize:
            loader_kwargs['num_workers'] = CPU_OPTIMIZE_NUM_WORKERS
            loader_kwargs['pin_memory'] = self.device != 'cpu'
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

        with torch.no_grad():
            for img_batch, label_batch in tqdm(loader, desc='Target model confidence'):
                img_batch = img_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                output = self.target_model(img_batch)
                confs = output.softmax(1)
                per_img_conf = torch.gather(
                    confs, 1, label_batch.unsqueeze(1)).squeeze(1)
                all_confs.append(per_img_conf.cpu())

        all_confs = torch.cat(all_confs)
        avg_conf = all_confs.mean().item()

        target_list = targets.cpu().tolist()
        conf_by_class_list = [['target', 'target_model_avg_conf']]
        for t in sorted(set(target_list)):
            mask = targets.cpu() == t
            cls_avg = all_confs[mask].mean().item()
            conf_by_class_list.append([t, cls_avg])

        return avg_conf, conf_by_class_list, all_confs.cpu().tolist()
