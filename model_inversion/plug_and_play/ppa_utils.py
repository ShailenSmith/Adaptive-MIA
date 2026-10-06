import argparse
import csv
import math
import random
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path
import pickle

import numpy as np
import torch
import torchvision.transforms as T
import wandb
from facenet_pytorch import InceptionResnetV1
from rtpt import RTPT
from torch.utils.data import TensorDataset

from Plug_and_Play_Attacks.attacks.final_selection import perform_final_selection
from Plug_and_Play_Attacks.attacks.optimize import Optimization
from Plug_and_Play_Attacks.datasets.custom_subset import ClassSubset
from Plug_and_Play_Attacks.metrics.classification_acc import ClassificationAccuracy
from Plug_and_Play_Attacks.metrics.fid_score import FID_Score
from Plug_and_Play_Attacks.metrics.prcd import PRCD
from Plug_and_Play_Attacks.utils.datasets import (create_target_dataset, get_facescrub_idx_to_class,
                            get_stanford_dogs_idx_to_class)
from Plug_and_Play_Attacks.utils.stylegan import create_image, load_discrimator, load_generator
from Plug_and_Play_Attacks.utils.wandb import *

from data_processing.datasets import get_datasets
from data_processing.data_augmentation import get_transforms

from utils.wandb_helpers import get_config


def safe_wandb_save(*args, **kwargs):
    """wandb.save() wrapper that swallows errors from a dead/broken wandb connection
    (e.g. BrokenPipeError after a long network outage) so a multi-hour attack run
    doesn't lose its already-disk-persisted results over a non-essential upload."""
    try:
        wandb.save(*args, **kwargs)
    except Exception as e:
        print(f"[wandb] save failed, continuing without upload: {e}")


def safe_wandb_log(*args, **kwargs):
    """wandb.log() wrapper that swallows errors from a dead/broken wandb connection,
    see safe_wandb_save."""
    try:
        wandb.log(*args, **kwargs)
    except Exception as e:
        print(f"[wandb] log failed, continuing without upload: {e}")


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing model inversion attack')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    parser.add_argument('--no_rtpt',
                        action='store_false',
                        dest="rtpt",
                        help='Disable RTPT')
    return parser

def create_initial_vectors(config, G, target_model, targets, device):
    with torch.no_grad():
        w = config.create_candidates(G, target_model, targets).cpu()
        if config.attack['single_w']:
            w = w[:, 0].unsqueeze(1)
        w_init = deepcopy(w)
        x = None
        V = None
    return w, w_init, x, V

def write_precision_list(filename, precision_list):
    filename = f"{filename}.csv"
    with open(filename, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for row in precision_list:
            wr.writerow(row)
    return filename

def log_attack_progress(loss,
                        target_loss,
                        discriminator_loss,
                        discriminator_weight,
                        mean_conf,
                        lr,
                        imgs=None,
                        captions=None):
    if imgs is not None:
        imgs = [
            wandb.Image(img.permute(1, 2, 0).numpy(), caption=caption)
            for img, caption in zip(imgs, captions)
        ]
        safe_wandb_log({
            'total_loss': loss,
            'target_loss': target_loss,
            'discriminator_loss': discriminator_loss,
            'discriminator_weight': discriminator_weight,
            'mean_conf': mean_conf,
            'learning_rate': lr,
            'samples': imgs
        })
    else:
        safe_wandb_log({
            'total_loss': loss,
            'target_loss': target_loss,
            'discriminator_loss': discriminator_loss,
            'discriminator_weight': discriminator_weight,
            'mean_conf': mean_conf,
            'learning_rate': lr
        })

def init_wandb_logging(optimizer, target_model_name, config):
    lr = optimizer.param_groups[0]['lr']
    optimizer_name = type(optimizer).__name__
    if not 'name' in config.wandb['wandb_init_args']:
        config.wandb['wandb_init_args'][
            'name'] = f'{optimizer_name}_{lr}_{target_model_name}'
    wandb_config = config.create_wandb_config()
    run = wandb.init(config=wandb_config, **config.wandb['wandb_init_args'])
    # wandb.save(args.config)
    return run

def intermediate_wandb_logging(optimizer, targets, confidences, loss,
                               target_loss, discriminator_loss,
                               discriminator_weight, mean_conf, imgs, idx2cls):
    lr = optimizer.param_groups[0]['lr']
    target_classes = [idx2cls[idx.item()] for idx in targets.cpu()]
    conf_list = [conf.item() for conf in confidences]
    if imgs is not None:
        img_captions = [
            f'{target} ({conf:.4f})'
            for target, conf in zip(target_classes, conf_list)
        ]
        log_attack_progress(loss,
                            target_loss,
                            discriminator_loss,
                            discriminator_weight,
                            mean_conf,
                            lr,
                            imgs,
                            captions=img_captions)
    else:
        log_attack_progress(loss, target_loss, discriminator_loss,
                            discriminator_weight, mean_conf, lr)

def log_nearest_neighbors(imgs, targets, eval_model, model_name, dataset,
                          img_size, seed):
    # Find closest training samples to final results
    evaluater = DistanceEvaluation(eval_model, None, img_size, None, dataset,
                                   seed)
    closest_samples, distances = evaluater.find_closest_training_sample(
        imgs, targets)
    closest_samples = [
        wandb.Image(img.permute(1, 2, 0).cpu().numpy(),
                    caption=f'distance={d:.4f}')
        for img, d in zip(closest_samples, distances)
    ]
    safe_wandb_log({f'closest_samples {model_name}': closest_samples})

def log_final_images(imgs, predictions, max_confidences, target_confidences,
                     idx2cls):
    wand_imgs = [
        wandb.Image(
            img.permute(1, 2, 0).numpy(),
            caption=
            f'pred={idx2cls[pred.item()]} ({max_conf:.2f}), target_conf={target_conf:.2f}'
        ) for img, pred, max_conf, target_conf in zip(
            imgs.cpu(), predictions, max_confidences, target_confidences)
    ]
    safe_wandb_log({'final_images': wand_imgs})

def final_wandb_logging(avg_correct_conf, avg_total_conf, acc_top1, acc_top5,
                        avg_dist_facenet, avg_dist_eval, fid_score, precision,
                        recall, density, coverage, total_avg_target_conf):
    safe_wandb_save('attacks/gradient_based.py')
    wandb.run.summary['correct_avg_conf'] = avg_correct_conf
    wandb.run.summary['total_avg_conf'] = avg_total_conf
    wandb.run.summary['evaluation_acc@1'] = acc_top1
    wandb.run.summary['evaluation_acc@5'] = acc_top5
    wandb.run.summary['avg_dist_facenet'] = avg_dist_facenet
    wandb.run.summary['avg_dist_evaluation'] = avg_dist_eval
    wandb.run.summary['fid_score'] = fid_score
    wandb.run.summary['precision'] = precision
    wandb.run.summary['recall'] = recall
    wandb.run.summary['density'] = density
    wandb.run.summary['coverage'] = coverage
    wandb.run.summary['WITHIN_avg_target_conf'] = total_avg_target_conf

    # # pickle targets and finaL_targets, then save them in wandb
    # targets = targets.cpu().numpy()
    # final_targets = final_targets.cpu().numpy()
    # targets = targets.tolist()
    # final_targets = final_targets.tolist()

    # # make sure targets and final_targets directories exist
    # targets_dir_path = "model_inversion/plug_and_play/results/targets"
    # final_targets_dir_path = "model_inversion/plug_and_play/results/final_targets"
    # os.mkdir(targets_dir_path, exist_ok=True)
    # os.mkdir(final_targets_dir_path, exist_ok=True)
    # targets_path = targets_dir_path + f'/{wandb.run.id}.pkl'
    # final_targets_path = final_targets_dir_path + f'/{wandb.run.id}.pkl'
    # with open(targets_path, 'wb') as f:
    #     pickle.dump(targets, f)
    # with open(final_targets_path, 'wb') as f:
    #     pickle.dump(final_targets, f)
    # wandb.save(targets_path)
    # wandb.save(final_targets_path)

    wandb.finish()
