import argparse
import csv
import math
import random
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import wandb
import os
from facenet_pytorch import InceptionResnetV1
from rtpt import RTPT
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

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
from model_inversion.plug_and_play.ppa_evaluate import ppa_evaluate
from model_inversion.plug_and_play.ppa_utils import *


def attack(config, target_dataset, target_model, evaluation_model, target_config, wandb_run = None):
    ####################################
    #        Attack Preparation        #
    ####################################

    # Set devices
    torch.set_num_threads(24)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # # Define and parse attack arguments
    # parser = create_parser()
    # config, args = parse_arguments(parser)

    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load idx to class mappings
    idx_to_class = None
    if config.dataset.lower() == 'facescrub':
        idx_to_class = get_facescrub_idx_to_class()
    elif config.dataset.lower() == 'stanford_dogs':
        idx_to_class = get_stanford_dogs_idx_to_class()
    else:

        class KeyDict(dict):

            def __missing__(self, key):
                return key

        idx_to_class = KeyDict()

    # Load pre-trained StyleGan2 components
    G = load_generator(config.stylegan_model, device)
    D = load_discrimator(config.stylegan_model, device)
    
    G.img_channels = target_dataset[0][0].shape[0] # Extracting the number of channels from the dataset
    
    num_ws = G.num_ws

    # target_dataset = config.get_target_dataset() #! Passed in as argument instead of loading here
    target_model_name = target_model.name
    target_model_num_classes = target_model.num_classes

    # Distribute models
    target_model = torch.nn.DataParallel(target_model, device_ids=gpu_devices)
    synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    discriminator = torch.nn.DataParallel(D, device_ids=gpu_devices)

    # Reinstanciate model variables
    target_model.name = target_model_name
    synthesis.num_ws = num_ws
    target_model.num_classes = target_model_num_classes
    
    
    # Load basic attack parameters
    num_epochs = config.attack['num_epochs']
    batch_size_single = config.attack['batch_size']
    batch_size = config.attack['batch_size'] * max(torch.cuda.device_count(), 1) #! Changed this to max(n, 1) to make it work with cpu.
    
    
    config.model = target_model #! Instead of running their load model, we pass it directly 
    
    targets = config.create_target_vector()

    # Create initial style vectors
    w, w_init, x, V = create_initial_vectors(config, G, target_model, targets,
                                             device)
    del G

    # Initialize wandb logging
    if config.logging:
        optimizer = config.create_optimizer(params=[w])
        # wandb_run = init_wandb_logging(optimizer, target_model_name, config) #! wandb_run is passed as argument instead
        
        run_id = wandb_run.id

    # Print attack configuration
    print(
        f'Start attack against {target_model.name} optimizing w with shape {list(w.shape)} ',
        f'and targets {dict(Counter(targets.cpu().numpy()))}.')
    print(f'\nAttack parameters')
    for key in config.attack:
        print(f'\t{key}: {config.attack[key]}')
    print(
        f'Performing attack on {torch.cuda.device_count()} gpus and an effective batch size of {batch_size} images.'
    )

    # Initialize RTPT
    rtpt = None
    # if args.rtpt: #! TODO: Maybe worth enabling as an option
    #     max_iterations = math.ceil(w.shape[0] / batch_size) \
    #         + int(math.ceil(w.shape[0] / (batch_size * 3))) \
    #         + 2 * int(math.ceil(config.final_selection['samples_per_target'] * len(set(targets.cpu().tolist())) / (batch_size * 3))) \
    #         + 2 * len(set(targets.cpu().tolist()))
    #     rtpt = RTPT(name_initials='LS',
    #                 experiment_name='Model_Inversion',
    #                 max_iterations=max_iterations)
    #     rtpt.start()

    # Log initial vectors
    if config.logging:
        Path("results").mkdir(parents=True, exist_ok=True)
        init_w_path = f"model_inversion/plug_and_play/results/init_w/{run_id}.pt"
        torch.save(w.detach(), init_w_path)
        safe_wandb_save(init_w_path)

    # Create attack transformations
    attack_transformations = config.create_attack_transformations()
    
    
    

    ####################################
    #         Attack Iteration         #
    ####################################
    optimization = Optimization(target_model, synthesis, discriminator,
                                attack_transformations, num_ws, config)

    # Collect results
    w_optimized = []

    # Prepare batches for attack
    for i in range(math.ceil(w.shape[0] / batch_size)):
        w_batch = w[i * batch_size:(i + 1) * batch_size].to(device)
        targets_batch = targets[i * batch_size:(i + 1) * batch_size].to(device)
        print(
            f'\nOptimizing batch {i+1} of {math.ceil(w.shape[0] / batch_size)} targeting classes {set(targets_batch.cpu().tolist())}.'
        )

        # Run attack iteration
        torch.cuda.empty_cache()
        w_batch_optimized = optimization.optimize(w_batch, targets_batch,
                                                  num_epochs).detach().cpu()

        if rtpt:
            num_batches = math.ceil(w.shape[0] / batch_size)
            rtpt.step(subtitle=f'batch {i+1} of {num_batches}')

        # Collect optimized style vectors
        w_optimized.append(w_batch_optimized)

    # Concatenate optimized style vectors
    w_optimized_unselected = torch.cat(w_optimized, dim=0)
    torch.cuda.empty_cache()
    del discriminator

    ####################################
    #          Filter Results          #
    ####################################

    # Filter results
    if config.final_selection:
        print(
            f'\nSelect final set of max. {config.final_selection["samples_per_target"]} ',
            f'images per target using {config.final_selection["approach"]} approach.'
        )
        final_w, final_targets = perform_final_selection(
            w_optimized_unselected,
            synthesis,
            config,
            targets,
            target_model,
            device=device,
            batch_size=batch_size * 10, #? Why multiplied by 10?
            **config.final_selection,
            rtpt=rtpt)
        print(f'Selected a total of {final_w.shape[0]} final images ',
              f'of target classes {set(final_targets.cpu().tolist())}.')
    else:
        final_targets, final_w = targets, w_optimized_unselected
        
    del target_model

    # Log optimized vectors and target tensors
    if config.logging:
        optimized_w_path = f"model_inversion/plug_and_play/results/optimized_w/{run_id}.pt"
        torch.save(w_optimized_unselected.detach(), optimized_w_path)
        safe_wandb_save(optimized_w_path)  # Log selected vectors

        optimized_w_path_selected = f"model_inversion/plug_and_play/results/optimized_w_selected/{run_id}.pt"
        torch.save(final_w.detach(), optimized_w_path_selected)
        safe_wandb_save(optimized_w_path_selected)

        os.makedirs("model_inversion/plug_and_play/results/targets", exist_ok=True)
        targets_path = f"model_inversion/plug_and_play/results/targets/{run_id}.pt"
        torch.save(targets.detach(), targets_path)
        safe_wandb_save(targets_path)

        os.makedirs("model_inversion/plug_and_play/results/final_targets", exist_ok=True)
        final_targets_path = f"model_inversion/plug_and_play/results/final_targets/{run_id}.pt"
        torch.save(final_targets.detach(), final_targets_path)
        safe_wandb_save(final_targets_path)

    # # Log and save final images
    # if config.logging:
    #     attack_dataset = TensorDataset(final_w, final_targets)
    #     final_imgs = []

    #     for step, (w_batch, target_batch) in enumerate(
    #             DataLoader(attack_dataset, batch_size=batch_size, shuffle=False)):
    #         w_batch, target_batch = w_batch.to(device), target_batch.to(device)
    #         img_batch = create_image(w_batch,
    #                                     synthesis,
    #                                     config.attack_center_crop,
    #                                     resize=299,
    #                                     batch_size=batch_size)
    #         img_batch = img_batch.detach().cpu()
    #         final_imgs.append(img_batch)

    #     final_imgs = torch.cat(final_imgs, dim=0).detach().cpu()
    #     # make sure attack_images directory exists
    #     attack_images_dir = "model_inversion/plug_and_play/results/final_attack_images"
    #     os.makedirs(attack_images_dir, exist_ok=True)
    #     final_imgs_path = f"{attack_images_dir}/{run_id}.pt"
    #     with open(final_imgs_path, 'wb') as f:
    #         torch.save(final_imgs, f)
    #     print(f"Saved final attack images to {final_imgs_path}")
    #     wandb.save(final_imgs_path)

    # del attack_dataset
    # del final_imgs

    avg_target_conf_dict = getattr(optimization, 'conf_dict', None)

    # Save per-iteration target model confidence values (same values printed to stdout)
    if config.logging:
        iter_conf_log = getattr(optimization, 'iter_conf_log', None)
        if iter_conf_log:
            os.makedirs("model_inversion/plug_and_play/results/iter_target_conf", exist_ok=True)
            iter_conf_path = f"model_inversion/plug_and_play/results/iter_target_conf/{run_id}.csv"
            with open(iter_conf_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['batch', 'iteration', 'target_classes', 'mean_conf', 'grad_cosine_sim'])
                writer.writerows(iter_conf_log)
            safe_wandb_save(iter_conf_path)

    #! BEGIN alignment score addition (see model_inversion/metrics/mi_gradient_alignment.py) - safe to delete to revert
    # Save per-iteration gradient-manifold alignment scores, if computed in-loop (attack.alignment_score.compute)
    if config.logging:
        alignment_log = getattr(optimization, 'alignment_log', None)
        if alignment_log:
            os.makedirs("model_inversion/plug_and_play/results/alignment_score", exist_ok=True)
            alignment_score_path = f"model_inversion/plug_and_play/results/alignment_score/{run_id}.csv"
            with open(alignment_score_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['batch', 'iteration', 'target_classes', 'sample', 'alignment_score'])
                writer.writerows(alignment_log)
            safe_wandb_save(alignment_score_path)

    # Save raw (w, grad_x) pairs needed to compute the alignment score post-hoc. Collected
    # unconditionally during optimization regardless of attack.alignment_score.compute, so this
    # is always logged (subject only to the same config.logging master switch as every other
    # artifact saved in this function).
    if config.logging:
        alignment_raw_log = getattr(optimization, 'alignment_raw_log', None)
        if alignment_raw_log:
            os.makedirs("model_inversion/plug_and_play/results/alignment_raw", exist_ok=True)
            alignment_raw_path = f"model_inversion/plug_and_play/results/alignment_raw/{run_id}.pt"
            torch.save(alignment_raw_log, alignment_raw_path)
            safe_wandb_save(alignment_raw_path)
    #! END alignment score addition

    ppa_evaluate(
        w_optimized_unselected=w_optimized_unselected,
        final_w=final_w,
        final_targets=final_targets,
        evaluation_model=evaluation_model,
        synthesis=synthesis,
        config=config,
        target_config=target_config,
        targets=targets,
        device=device,
        gpu_devices=gpu_devices,
        idx_to_class=idx_to_class,
        batch_size=batch_size,
        batch_size_single=batch_size_single,
        run_id=run_id,
        target_dataset=target_dataset,
        avg_target_conf_dict=avg_target_conf_dict,
    )
