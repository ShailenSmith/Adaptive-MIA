
#! WIP -- don't push to public yet
import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Plug_and_Play_Attacks.metrics.accuracy import Accuracy, AccuracyTopK

# Only used when cpu_optimize=True -- see model_inversion.metrics.mi_distance_metrics.
CPU_OPTIMIZE_NUM_WORKERS = min(4, os.cpu_count() or 1)


class ClassificationAccuracy():

    def __init__(self, evaluation_network, device='cuda:0'):
        self.evaluation_network = evaluation_network
        self.device = device

    def compute_acc(self,
                    dataset,
                    targets,
                    batch_size=64,
                    rtpt=None,
                    cpu_optimize=False):
        """
        cpu_optimize: if True, load images with worker processes + pinned memory
            instead of serially on the main process (doesn't affect the result).
            Default False reproduces the exact prior behavior. Same knob as
            DistanceEvaluation.compute_dist / evaluate_attack's cpu_optimize.
        """
        self.evaluation_network.eval()
        self.evaluation_network.to(self.device)
        acc_top1 = Accuracy()
        acc_top5 = AccuracyTopK(k=5)
        predictions = []
        correct_top5 = []
        correct_confidences = []
        total_confidences = []
        maximum_confidences = []

        max_iter = math.ceil(len(dataset) / batch_size)

        loader_kwargs = {}
        if cpu_optimize:
            loader_kwargs['num_workers'] = CPU_OPTIMIZE_NUM_WORKERS
            loader_kwargs['pin_memory'] = self.device != 'cpu'
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

        with torch.no_grad():
            for step, (img_batch, target_batch) in tqdm(enumerate(loader), desc='Accuracy', total=max_iter):
                img_batch, target_batch = img_batch.to(self.device), target_batch.to(self.device)
                
                output = self.evaluation_network(img_batch)

                acc_top1.update(output, target_batch)
                acc_top5.update(output, target_batch)

                pred = torch.argmax(output, dim=1)
                predictions.append(pred)
                confidences = output.softmax(1)
                target_confidences = torch.gather(confidences, 1,
                                                  target_batch.unsqueeze(1))
                correct_confidences.append(
                    target_confidences[pred == target_batch])
                total_confidences.append(target_confidences)
                maximum_confidences.append(torch.max(confidences, dim=1)[0])

                _, top5_preds = torch.topk(output, k=min(5, output.shape[1]), dim=1)
                in_top5 = (top5_preds == target_batch.unsqueeze(1)).any(dim=1)
                correct_top5.append(in_top5.cpu())

            acc_top1 = acc_top1.compute_metric()
            acc_top5 = acc_top5.compute_metric()
            correct_confidences = torch.cat(correct_confidences, dim=0)
            avg_correct_conf = correct_confidences.mean().cpu().item()
            confidences = torch.cat(total_confidences, dim=0).cpu()
            confidences = torch.flatten(confidences)
            maximum_confidences = torch.cat(maximum_confidences,
                                            dim=0).cpu().tolist()
            avg_total_conf = torch.cat(total_confidences,
                                       dim=0).mean().cpu().item()
            predictions = torch.cat(predictions, dim=0).cpu()
            correct_top5 = torch.cat(correct_top5, dim=0)

            # Compute class-wise precision
            target_list = targets.cpu().tolist()
            precision_list = [['target', 'mean_conf', 'acc_top1', 'acc_top5']]
            for t in set(target_list):
                mask = torch.where(targets == t, True, False).cpu()
                conf_masked = confidences[mask]
                top1_acc = torch.sum(predictions[mask] == t) / torch.sum(
                    targets == t)
                top5_acc = correct_top5[mask].float().mean()
                precision_list.append(
                    [t, conf_masked.mean().item(),
                     top1_acc.cpu().item(), top5_acc.cpu().item()])
            confidences = confidences.tolist()
            predictions = predictions.tolist()

            if rtpt:
                rtpt.step(
                    subtitle=
                    f'Classification Evaluation step {step} of {max_iter}')

        return acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, \
            confidences, maximum_confidences, precision_list
