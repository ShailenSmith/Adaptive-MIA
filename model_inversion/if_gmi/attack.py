import csv
import math
import os
import shutil
import random
import traceback
import pdb
import psutil
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import time
import torch
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1
from rtpt import RTPT
from torch.utils.data import TensorDataset

import sys
import os

import copy

# Add the IF-GMI folder to the Python path
# sys.path.insert(0, 'IF-GMI')

from IF_GMI.attacks.optimize import Optimization
from IF_GMI.datasets.custom_subset import ClassSubset
from IF_GMI.metrics.classification_acc import ClassificationAccuracy
from IF_GMI.metrics.fid_score import FID_Score
from IF_GMI.metrics.distance_metrics import DistanceEvaluation
from IF_GMI.metrics.prdc import PRDC
from IF_GMI.utils.logger import *
from IF_GMI.utils.datasets import (get_facescrub_idx_to_class,
                                         get_stanford_dogs_idx_to_class)
from IF_GMI.utils.stylegan import crop_and_resize, load_generator
from IF_GMI.utils.logger import create_initial_vectors, save_dict_to_yaml

from data_processing.datasets import get_datasets
from data_processing.data_augmentation import get_transforms

import wandb

from IF_GMI.pkl2pth import change

def attack(config, target_dataset, target_model, evaluation_model, target_config, wandb_run = None):
    ####################################
    #        Attack Preparation        #
    ####################################
    
    # Record running time and occupied memory
    start_time = time.perf_counter()
    now_time = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
    init_mem = psutil.virtual_memory().free
    min_mem = init_mem
    
    rtpt = None

    # Set devices
    # torch.set_num_threads(24)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # Define and parse attack arguments
    # parser = create_parser()
    # config, args = parse_arguments(parser)
    layer_num = len(config.intermediate['steps'])

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

    # Load pre-trained StyleGan2 generator
    
    stylegan_name = os.path.basename(config.stylegan_model)
    stylegan_filetype = os.path.splitext(stylegan_name)[-1]
    
    if stylegan_filetype == ".pkl":
        new_path = os.path.splitext(config.stylegan_model)[0] + ".pth"
        
        if not os.path.exists(new_path):
            change(config.stylegan_model, new_path)
        
        config._config['stylegan_model']=new_path
    
    
    G = load_generator(config.stylegan_model)
    num_ws = G.num_ws

    # Load target model and dataset
    # target_model, target_name = config.create_target_model()
    target_model_name = target_model.name
    # target_dataset = config.get_target_dataset()

    # Distribute models in multiple GPUs
    target_model = torch.nn.DataParallel(target_model, device_ids=gpu_devices)
    synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    
    target_model.name = target_model_name
    synthesis.num_ws = num_ws

    # Load basic attack parameters
    batch_size_single = config.attack['batch_size']
    batch_size = config.attack['batch_size'] * len(gpu_devices)
    eval_batch_size = config.attack['eval_batch_size'] * len(gpu_devices)
    targets = config.create_target_vector()
    
    # set transformations for images
    crop_size = config.attack_center_crop
    target_transform = T.Compose([
        T.ToTensor(),
        T.Resize((299, 299), antialias=True), #! Why not 224 like plug and play?
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    full_training_dataset, _, _ = get_datasets(target_config, target_transform, None)

    # Load evaluation model Incv3
    # evaluation_model, eval_name = config.create_evaluation_model()
    evaluation_model = torch.nn.DataParallel(evaluation_model, device_ids=gpu_devices)
    evaluation_model.to(device)
    evaluation_model.eval()
    class_acc_evaluator = ClassificationAccuracy(evaluation_model,
                                                 layer_num=layer_num,
                                                 device=device)
    
    fid_evaluation = FID_Score(layer_num,
                                  device=device,
                                  crop_size=crop_size,
                                  batch_size=eval_batch_size, #batch_size * 3,
                                  dims=2048,
                                  num_workers=0,
                                  gpu_devices=gpu_devices)

    prdc = PRDC(layer_num,
                   device=device,
                   crop_size=crop_size,
                   batch_size=eval_batch_size, #batch_size * 3,
                   dims=2048,
                   num_workers=0,
                   gpu_devices=gpu_devices)

    # Load Inception-v3 evaluation model and remove final layer
    # evaluation_model_dist, _ = config.create_evaluation_model()

    # evaluation_model_dist = copy.deepcopy(evaluation_model)
    # if isinstance(evaluation_model_dist, torch.nn.DataParallel):
    #     evaluation_model_dist.module.model.fc = torch.nn.Sequential()  # This is the fix 
    # else:
    #     evaluation_model_dist.model.fc = torch.nn.Sequential() # This doesn't work
    #     evaluation_model_dist = torch.nn.DataParallel(evaluation_model_dist,
    #                                                 device_ids=gpu_devices)
        
    if isinstance(evaluation_model, torch.nn.DataParallel):
        evaluation_model_dist = evaluation_model.module.model.feature_extractor
    else:
        evaluation_model_dist = evaluation_model.model.feature_extractor

    evaluation_model_dist.to(device)
    evaluation_model_dist.eval()

    inception_dist = DistanceEvaluation(
        layer_num, evaluation_model_dist,
        299,
        config.attack_center_crop,
        target_dataset, config.seed)

    # Load FaceNet model for face recognition
    facenet = InceptionResnetV1(pretrained='vggface2')
    facenet = torch.nn.DataParallel(
        facenet, device_ids=gpu_devices)
    facenet.to(device)
    facenet.eval()

    facenet_dist = DistanceEvaluation(layer_num, facenet, 160,
                                              config.attack_center_crop,
                                              target_dataset, config.seed)

    ####################################
    #              Attack              #
    ####################################

    # Create initial style vectors
    w = create_initial_vectors(config, G, target_model, targets,
                                             device)
    # w = torch.load("initial_vec_if.pt")
    del G

    # Initialize logging
    result_path = config.path
    if config.logging:
        
        if wandb_run:
            run_id = wandb_run.id
        else:
            run_id = now_time
        
        init_w_path = f"{result_path}/init_w_{run_id}.pt"
        torch.save(w.detach(), init_w_path)


    # Print attack configuration: 打印攻击参数设置
    print(
        f'Start attack against {target_model.name} optimizing w with shape {list(w.shape)} ',
        f'and targets {dict(Counter(targets.cpu().numpy()))}.')
    print(f'\nAttack parameters')
    for key in config.attack:
        print(f'\t{key}: {config.attack[key]}')
    print(
        f'Performing attack on {len(gpu_devices)} gpus and an effective batch size of {batch_size} images.'
    )

    # Initialize RTPT
    # rtpt = None
    # if args.rtpt:
    #     max_iterations = math.ceil(w.shape[0] / batch_size) \
    #         + int(math.ceil(w.shape[0] / (batch_size * 3))) \
    #         + 2 * int(math.ceil(config.candidates['num_candidates'] * len(set(targets.cpu().tolist())) / (batch_size * 3))) \
    #         + 2 * len(set(targets.cpu().tolist()))
    #     rtpt = RTPT(name_initials='IF-GMI',
    #                 experiment_name='Model_Inversion_Attack',
    #                 max_iterations=max_iterations)
    #     rtpt.start()

    # Create attack transformations
    attack_transformations = config.create_attack_transformations()
    now_mem = psutil.virtual_memory().free
    print(f'free memory after attack preparation:{(now_mem / (1024**3)):.4f}GB')
    min_mem = min(now_mem, min_mem)

    optimization = Optimization(target_model, synthesis, attack_transformations, num_ws, config)

    # Prepare to collect results
    w_optimized_all = {i: [] for i in range(layer_num)}
    final_targets_all = []

    unique_targets = list(dict.fromkeys(targets.tolist())) # to preserve order of targets
    n_class_digits = len(str(max(unique_targets)))

    # iteratively compute each target class (reduce memory cost)
    for target_idx_in_list, target_id in enumerate(unique_targets):
        num_candidates = config.candidates['num_candidates']
        optimization.flush_imgs()

        now_mem = psutil.virtual_memory().free
        print(f'free memory before attacking {target_id}:{(now_mem / (1024**3)):.4f}GB')
        min_mem = min(now_mem, min_mem)

        # Prepare batches for attack
        for i in range(math.ceil(num_candidates / batch_size)):
            start_idx = target_idx_in_list * num_candidates + i * batch_size
            end_idx = min(start_idx + batch_size, (target_idx_in_list+1)*num_candidates)
            w_batch = w[start_idx:end_idx].cuda()
            targets_batch = targets[start_idx:end_idx].cuda()
            if target_id not in targets_batch.cpu().tolist():
                print(f"WARNING: target {target_id} not in batch {i+1} of {math.ceil(num_candidates / batch_size)}. Targets_batch: {targets_batch.cpu().tolist()}")
            print(
                f'\nOptimizing batch {i+1} of {math.ceil(num_candidates / batch_size)} targeting classes {set(targets_batch.cpu().tolist())}.'
            )

            # Run attack iteration
            torch.cuda.empty_cache()
            optimization.optimize(w_batch, targets_batch)

            # if rtpt:
            #     num_batches = math.ceil(w.shape[0] / batch_size)
            #     rtpt.step(
            #         subtitle=f'batch {i+1+target_idx_in_list*math.ceil(num_candidates / batch_size)} of {num_batches}')

        # Concatenate optimized style vectors
        w_optimized = optimization.intermediate_w
        imgs_optimized = optimization.intermediate_imgs
        for k, v in imgs_optimized.items():
            imgs_optimized[k] = torch.cat(v, dim=0)
        for k, v in w_optimized.items():
            w_optimized[k] = torch.cat(v, dim=0)
            w_optimized_all[k].append(w_optimized[k])

        torch.cuda.empty_cache()
        
        # record results
        target_list = targets[target_idx_in_list*num_candidates:(target_idx_in_list+1)*num_candidates]
        final_targets, final_w, final_imgs = target_list, w_optimized, imgs_optimized
        final_targets_all.append(final_targets)

        now_mem = psutil.virtual_memory().free
        print(f'free memory after attacking {target_id}: {(now_mem / (1024**3)):.4f}GB')
        min_mem = min(now_mem, min_mem)

        ####################################
        #         Attack Accuracy          #
        ####################################
        
        # Compute attack accuracy with evaluation model on all generated samples
        try:
            print('compute confidences for acc')
            for layer in range(layer_num):
                class_acc_evaluator.compute_confidences(
                    layer,
                    imgs_optimized[layer],
                    target_list,
                    config,
                    batch_size=eval_batch_size, #batch_size * 2,
                    resize=299,
                    rtpt=rtpt)

        except Exception:
            print(traceback.format_exc())

        ####################################
        #    FID Score and GAN Metrics     #
        ####################################
        target_list = target_list.cpu()
        try:
            training_dataset = ClassSubset(
                full_training_dataset,
                target_classes=torch.unique(target_list).cpu().tolist())
            for layer in range(layer_num):
                # create datasets
                attack_dataset = TensorDataset(
                    imgs_optimized[layer], target_list)
                attack_dataset.targets = target_list

                # compute FID score
                start_time = time.time()
                print(f'calculate fid for layer {layer}')
                fid_evaluation.set(training_dataset, attack_dataset)
                fid_evaluation.get_preds(layer, rtpt)
                print(f'FID score embedding time for layer {layer}: {time.time() - start_time:.4f} seconds')

                start_time = time.time()
                # compute precision, recall, density, coverage
                print(f'calculate prdc for layer {layer}')
                prdc.set(training_dataset, attack_dataset)
                prdc.compute_metric(
                    layer, int(target_list[0]), k=3, rtpt=rtpt)
                print(f'PRDC score embedding time for layer {layer}: {time.time() - start_time:.4f} seconds')


        except Exception:
            print(traceback.format_exc())

        ####################################
        #         Feature Distance         #
        ####################################
        try:
            print('calculate feature distance')
            for layer in range(layer_num):
                start_time = time.time()
                inception_dist.compute_dist(
                    layer,
                    imgs_optimized[layer],
                    target_list,
                    batch_size=eval_batch_size, #batch_size_single * 20,
                    rtpt=rtpt)
                print(f'Inception distance embedding time for layer {layer}: {time.time() - start_time:.4f} seconds')

            # Compute feature distance only for facial images
            is_face = target_config.dataset.face_dataset
            if is_face:
                for layer in range(layer_num):
                    start_time = time.time()
                    facenet_dist.compute_dist(
                        layer,
                        imgs_optimized[layer],
                        target_list,
                        batch_size=eval_batch_size, #batch_size_single * 10,
                        rtpt=rtpt)
                    print(f'FaceNet distance embedding time for layer {layer}: {time.time() - start_time:.4f} seconds')
        except Exception:
            print(traceback.format_exc())

        now_mem = psutil.virtual_memory().free
        print(f'free memory when evaluation {target_id}: {(now_mem / (1024**3)):.4f}GB')
        min_mem = min(now_mem, min_mem)

        if config.logging:
            for layer in range(layer_num):
                imgs_to_save = crop_and_resize(
                    imgs_optimized[layer],
                    crop_size=config.attack_center_crop,
                    resize=config.attack_resize
                ).cpu()
                class_dir = os.path.join(
                    result_path, f'layer{layer}', run_id,
                    str(target_id).zfill(n_class_digits))
                os.makedirs(class_dir, exist_ok=True)
                for i, img in enumerate(imgs_to_save):
                    save_image(img, os.path.join(class_dir, f'{i:04d}.png'), normalize=True)

        if config.logging_images:
            log_images(config, result_path, evaluation_model, target_id, layer_num, final_imgs, idx_to_class, run_id=run_id)

    print(f'maxima occupied memory:{((init_mem-min_mem) / (1024**3)):.4f}GB')

    # aggregate
    for k in range(layer_num):
        w_optimized_all[k] = torch.cat(
            w_optimized_all[k], dim=0)

    ####################################
    #          Finish Logging          #
    ####################################


    #! Log avg confs per layer and class
    conf_dict = optimization.conf_dict
    avg_conf_class_dict = {} # avg_conf_class_dict[class] = avg conf value
    for layer in conf_dict:
        avg_conf_class_dict[layer] = {}
        for class_id, conf_values in conf_dict[layer].items():
            avg_conf_class_dict[layer][class_id] = sum(conf_values) / len(conf_values)
        # Log avg confs
        # print(f'Layer {layer} average confidences: {avg_conf_class_dict}')
        layer_avg_conf = sum(avg_conf_class_dict[layer].values()) / len(avg_conf_class_dict[layer].values())
        wandb.log({f"WITHIN_avg_conf_layer{layer}": layer_avg_conf})


    if config.logging:
        print('Finishing attack, logging results and creating sample images.')

        # save optimized noise vectors
        os.makedirs(f"{result_path}/optimized_w", exist_ok=True)
        optimized_w_path = f"{result_path}/optimized_w/{run_id}.pt"
        torch.save(w_optimized_all, optimized_w_path)
        if wandb_run:
            wandb.save(optimized_w_path)

        # save list of all targets
        final_targets_dir = os.path.join(result_path, 'final_targets')
        os.makedirs(final_targets_dir, exist_ok=True)
        final_targets_path = os.path.join(final_targets_dir, f'{run_id}.pt')
        torch.save(targets.detach(), final_targets_path)

        if wandb_run:
            wandb.save(final_targets_path)

        #! BEGIN grad_x / alignment logging addition (see IF_GMI/attacks/optimize.py) - safe to delete to revert
        # Save per-iteration target model confidence values (same values printed to stdout),
        # mirrors model_inversion/plug_and_play/attack.py's iter_target_conf logging.
        iter_conf_log = getattr(optimization, 'iter_conf_log', None)
        if iter_conf_log:
            os.makedirs(f"{result_path}/iter_target_conf", exist_ok=True)
            iter_conf_path = f"{result_path}/iter_target_conf/{run_id}.csv"
            with open(iter_conf_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['batch', 'layer', 'iteration', 'target_classes', 'mean_conf', 'grad_cosine_sim'])
                writer.writerows(iter_conf_log)
            if wandb_run:
                wandb.save(iter_conf_path)

        # Save raw (w, mid_vector, grad_x) tuples needed to compute the gradient-manifold
        # alignment score post-hoc; see the caveat in IF_GMI/attacks/optimize.py about why
        # model_inversion/metrics/mi_gradient_alignment.py can't consume this log unmodified.
        alignment_raw_log = getattr(optimization, 'alignment_raw_log', None)
        if alignment_raw_log:
            os.makedirs(f"{result_path}/alignment_raw", exist_ok=True)
            alignment_raw_path = f"{result_path}/alignment_raw/{run_id}.pt"
            torch.save(alignment_raw_log, alignment_raw_path)
            if wandb_run:
                wandb.save(alignment_raw_path)
        #! END grad_x / alignment logging addition

        # get main metrics
        best_layer_result = [float('-inf')]
        for i in range(layer_num):
            acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.get_compute_result(i,
                                                                                                                                                                                targets)
            if acc_top1 > best_layer_result[0]:
                best_layer_result = [acc_top1, acc_top5, predictions, avg_correct_conf,
                                     avg_total_conf, target_confidences, maximum_confidences, precision_list, i]
            print(
                f'Evaluation of {w_optimized_all[0].shape[0]} images on Inception-v3 and layer {i}: \taccuracy@1={acc_top1:4f}',
                f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
            )
        best_layer = best_layer_result[-1]
        print(
            f'Evaluation of {w_optimized_all[0].shape[0]} images on Inception-v3 and best layer is {best_layer}!'
        )

        # copy best layer ImageFolder to best_layer/, then remove per-layer folders
        src = os.path.join(result_path, f'layer{best_layer}', run_id)
        dst = os.path.join(result_path, 'best_layer', run_id)
        if os.path.exists(src):
            shutil.copytree(src, dst)
        for layer in range(layer_num):
            layer_dir = os.path.join(result_path, f'layer{layer}', run_id)
            if os.path.exists(layer_dir):
                shutil.rmtree(layer_dir)
        if wandb_run and os.path.exists(dst):
            artifact = wandb.Artifact(name=f'reconstructions_{run_id}', type='dataset')
            artifact.add_dir(dst, name='best_layer')
            wandb_run.log_artifact(artifact)

        # save precision list
        try:
            os.makedirs(f'{result_path}/precision_list_best', exist_ok=True)
            precision_list_best_filename = write_precision_list(
                f'{result_path}/precision_list_best/{run_id}',
                best_layer_result[-2])
            
            if wandb_run:
                wandb.save(precision_list_best_filename)
        except Exception as e:
            print(e)

        # save main metrics and confidence histogram
        if wandb_run:
            [acc_top1, acc_top5, predictions, avg_correct_conf, 
                avg_total_conf, target_confidences, maximum_confidences, precision_list, i] = best_layer_result
            wandb.log({
                "evaluation_acc@1": acc_top1,
                "evaluation_acc@5": acc_top5,
                # "predictions": predictions,
                "correct_avg_conf": avg_correct_conf,
                "total_avg_conf": avg_total_conf,
                # "target_confidences": target_confidences,
                # "maximum_confidences": maximum_confidences,
                # "precision_list": precision_list,
                "Best Iteration": i,
                "WITHIN_avg_conf_best_layer": sum(avg_conf_class_dict[i].values()) / len(avg_conf_class_dict[i].values()),
                })
            conf_bins = 30
            plt = plot_histogram(target_confidences, conf_bins, 'Distribution of Maximum Confidences',
                             'Confidence', 'Frequency', color='lightgreen')
            wandb.log({"Confidence Histogram": wandb.Image(plt)})

        # save fid and prdc
        for i in range(layer_num):
            fid_score = fid_evaluation.compute_fid(i)
            precision, recall, density, coverage = prdc.get_prdc(i)
            print(f'Evaluation metrics of layer {i}:')
            print(
                f'\tFID score computed on {w_optimized_all[0].shape[0]} attack samples and {config.dataset}: {fid_score:.4f}'
            )
            print(
                f' \tPrecision: {precision:.4f}, Recall: {recall:.4f}, Density: {density:.4f}, Coverage: {coverage:.4f}'
            )
            if wandb_run and i == best_layer:
                wandb.log({
                        "fid_score": fid_score,
                        "precision": precision,
                        "recall": recall,
                        "density": density,
                        "coverage": coverage})  
        print('\n')
        
        # save Inception feature distance
        mean_distances_lists = []
        for i in range(layer_num):
            avg_dist_inception, mean_distances_list = inception_dist.get_eval_dist(
                i)
            mean_distances_lists.append(mean_distances_list)
            print(f'Mean Distance on Inception-v3 and layer {i}: ',
                  avg_dist_inception.cpu().item())
            if wandb_run and i == best_layer:
                wandb.log({
                    f"avg_dist_evaluation": avg_dist_inception.cpu().item()
                }) 
        try:
            os.makedirs(f'{result_path}/distance_inceptionv3_list_best', exist_ok=True)
            inception_distance_filename = write_precision_list(
                # f'{result_path}/distance_inceptionv3_list_best_{run_id}',
                os.path.join(result_path, 'distance_inceptionv3_list_best', run_id),
                mean_distances_lists[best_layer])
            
            if wandb_run:
                wandb.save(inception_distance_filename)
        except Exception as e:
            print(e)
        
        # save FaceNet feature distance
        if is_face:
            mean_distances_lists = []
            for i in range(layer_num):
                avg_dist_facenet, mean_distances_list = facenet_dist.get_eval_dist(
                    i)
                mean_distances_lists.append(mean_distances_list)
                print(f'Mean Distance on FaceNet and layer {i}: ',
                    avg_dist_facenet.cpu().item())
                
                if wandb_run and i == best_layer:
                    wandb.log({
                        f"avg_dist_facenet": avg_dist_facenet.cpu().item()
                    })   
            try:
                os.makedirs(f'{result_path}/distance_facenet_list_best', exist_ok=True)
                facenet_distance_filename = write_precision_list(
                    os.path.join(result_path, 'distance_facenet_list_best', run_id),
                    # f'{result_path}/distance_facenet_list_best_{run_id}',
                    mean_distances_lists[best_layer]
                    )
                if wandb_run:
                    wandb.save(facenet_distance_filename)      
            except Exception as e:
                print(e)

        #! save AvgTargetConf per class
        try:
            for layer in avg_conf_class_dict:
                avg_conf_class_with_ids = ['target', 'avg_target_conf'] + [[class_id, avg_conf] for class_id, avg_conf in avg_conf_class_dict[layer].items()]
                os.makedirs(f'{result_path}/avg_target_conf_per_class_layer{layer}', exist_ok=True)
                avg_target_conf_filename = write_precision_list(
                    os.path.join(result_path, f'avg_target_conf_per_class_layer{layer}', run_id),
                    avg_conf_class_with_ids
                )
                if layer == best_layer:
                    os.makedirs(f'{result_path}/avg_target_conf_per_class_best_layer', exist_ok=True)
                    bestlayer_avg_target_conf_filename = write_precision_list(
                        os.path.join(result_path, f'avg_target_conf_per_class_best_layer', run_id),
                        avg_conf_class_with_ids
                    )
                    if wandb_run:
                        wandb.save(bestlayer_avg_target_conf_filename)
                if wandb_run:
                    wandb.save(avg_target_conf_filename)
        except Exception as e:
            print(e)

        if wandb_run:
            wandb.finish()

        # save time
        end_time = time.perf_counter()
        with open(f'{result_path}/time.txt', 'w') as file:
            file.write(f'running time: {end_time-start_time:.4f} seconds')

    if rtpt:
        rtpt.step(subtitle=f'Finishing up')

def plot_histogram(data, bins, title, xlabel, ylabel, color):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', color=color, alpha=0.8)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.tight_layout()
    return plt
