
#! WIP -- don't push to public yet

import copy
import os
import traceback

import numpy as np
import torch
import torchvision.transforms as T
import wandb
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1
from Plug_and_Play_Attacks.datasets.custom_subset import ClassSubset

from model_inversion.metrics.mi_classification_acc import ClassificationAccuracy
from model_inversion.metrics.mi_fid_score import FID_Score
from model_inversion.metrics.mi_prdc import PRDC
from model_inversion.metrics.mi_distance_metrics import DistanceEvaluation
from model_inversion.metrics.mi_target_model_conf import TargetModelConfidence

from data_processing.datasets import get_datasets
from data_processing.data_augmentation import get_transforms

from model_inversion.plug_and_play.ppa_utils import (
    write_precision_list, log_final_images, log_nearest_neighbors,
)

# Metric groups selectable in the run_wandb_evaluate.py picker. "Alignment score" is
# listed for forward compatibility but is not yet wired up to any computation here.
METRIC_GROUPS = [
    "Eval model accuracy / confidence",
    "Robust eval model accuracy / confidence",
    "Facenet distance",
    "Inception distance",
    "Robust Inception distance",
    "Target model confidence",
    "FID",
    "PRDC",
    "Alignment score",
    "sample image logging (v1)",
    "sample image logging (v2)",
]

# Number of images per column in the "sample image logging (v2)" percentile comparison
# plot. Shared with run_wandb_compare_samples.py so combined figures line up row-for-row.
SAMPLE_IMAGE_V2_NUM_PER_COL = 10


def eps_metric_suffix(eps):
    """Format an adversarial-training epsilon value as a metric-name suffix, e.g.
    0.031 -> '_eps.031', 0.0025 -> '_eps.0025'. Shared with run_wandb_evaluate.py and
    scripts/backfill_robust_eval_eps.py so robust-evaluation metric names stay consistent
    across past and future runs."""
    return f"_eps{str(eps).lstrip('0')}"


def select_log_classes(targets, num_log_classes=10):
    """Pick which classes to build sample-image plots for: the first and last
    num_log_classes/2 classes in attack order. Shared by "sample image logging (v1)"
    and "(v2)" so both pick the same classes."""
    all_labels = list(dict.fromkeys(targets.tolist()))  # preserve attack order
    return set(all_labels[:num_log_classes // 2] + all_labels[-(num_log_classes // 2):])


def build_percentile_columns(attack_img_dataset, targets, eval_model_confidences,
                             target_model_confidences, class_label,
                             num_imgs_per_col=SAMPLE_IMAGE_V2_NUM_PER_COL):
    """Pick one reconstructed image of `class_label` per percentile of eval-model
    confidence (left column) and of target-model confidence (right column), for the
    "sample image logging (v2)" plot. Percentiles are computed within this class only.

    Returns:
        dict with 'class_label', 'num_imgs_per_col', 'percentiles' (list[float]), and
        'left'/'right' entries, each {'images': FloatTensor [num_imgs_per_col, C, H, W]
        in [0, 1], 'eval_conf': list[float], 'target_conf': list[float], 'targets': list[int]}.
    """
    targets_arr = np.asarray(targets.tolist() if torch.is_tensor(targets) else targets)
    class_indices = np.nonzero(targets_arr == class_label)[0]

    percentiles = np.linspace(100, 100 / num_imgs_per_col, num_imgs_per_col)
    eval_conf_arr = np.asarray(eval_model_confidences)[class_indices]
    target_conf_arr = np.asarray(target_model_confidences)[class_indices]

    def pick_indices(values):
        return [int(class_indices[np.argmin(np.abs(values - np.percentile(values, p)))])
                for p in percentiles]

    def build_column(indices):
        images = torch.stack(
            [(attack_img_dataset[i][0].clamp(-1, 1) + 1) / 2 for i in indices])
        return {
            'images': images,
            'eval_conf': [float(eval_model_confidences[i]) for i in indices],
            'target_conf': [float(target_model_confidences[i]) for i in indices],
            'targets': [int(targets[i]) for i in indices],
        }

    return {
        'class_label': int(class_label),
        'num_imgs_per_col': num_imgs_per_col,
        'percentiles': percentiles.tolist(),
        'left': build_column(pick_indices(eval_conf_arr)),
        'right': build_column(pick_indices(target_conf_arr)),
    }


def render_percentile_column(axes, column, percentiles, header=None):
    """Render one column (a sequence of Axes, one per row) of a percentile comparison plot.

    `column` is the 'left' or 'right' dict returned by `build_percentile_columns`. `header`
    (bold) is only set as the title of the top-most Axes, to label the column.
    """
    for row, ax in enumerate(axes):
        img = column['images'][row]
        eval_conf = column['eval_conf'][row]
        target_conf = column['target_conf'][row]
        pct = percentiles[row]

        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'{pct:.0f}th pct\neval={eval_conf:.2f} target={target_conf:.2f}', fontsize=8)
        if row == 0 and header:
            ax.set_title(header, fontsize=8, fontweight='bold')


def render_reference_row(axes, ref_imgs, label=None):
    """Render real reference images (each a FloatTensor [C, H, W] in [0, 1]) into `axes`,
    a sequence of Axes evenly spanning the width of the percentile columns they sit above
    (e.g. Axes created from a nested GridSpec subdividing the slice covering those
    columns), so the group of reference images appears centered above them as a whole.
    `ref_imgs` may have fewer entries than `axes`; leftover Axes are left blank. `label`
    (bold), if given, is set as the title of the middle Axes.
    """
    ref_imgs = list(ref_imgs) + [None] * (len(axes) - len(ref_imgs))
    for ax, ref_img in zip(axes, ref_imgs):
        if ref_img is not None:
            ax.imshow(ref_img.permute(1, 2, 0).numpy())
        ax.set_xticks([])
        ax.set_yticks([])
    if label:
        axes[len(axes) // 2].set_title(label, fontsize=9, fontweight='bold')


def evaluate_attack(evaluation_model,
                    target_config,
                    attack_img_dataset,
                    targets,
                    device,
                    gpu_devices,
                    idx_to_class,
                    batch_size,
                    batch_size_single,
                    run_id,
                    target_dataset,
                    result_path,
                    wandb_run=None,
                    target_model=None,
                    robust_evaluation_models=None,
                    rtpt=None,
                    overwrite_run=False,
                    metric_groups=None,
                    keep_existing_metrics=False,
                    cpu_optimize=False):
    """Evaluate a set of pre-generated attack images against all standard MI metrics.
    No seed is set here, should be set in the calling function if needed for reproducibility of distance metrics.

    Args:
        evaluation_model: classifier with a `.model.feature_extractor` attribute
        target_config: Hydra DictConfig for the target dataset/training setup
        attack_img_dataset: ImageFolder (or any Dataset) of attack images at 299x299,
            normalized consistently with the training dataset. Its order must match
            `targets`. Its `.targets` attribute will be overwritten with `targets`.
        targets: Tensor [N] of original class IDs corresponding to each image
        device: 'cuda' or 'cpu'
        gpu_devices: list of GPU device IDs
        idx_to_class: dict mapping class index -> class name
        batch_size: base batch size (multiplied by gpu count for DataParallel)
        batch_size_single: single-GPU batch size (for distance evaluation)
        run_id: string identifier for saved files
        target_dataset: full training dataset (unfiltered) used for nearest-neighbor logging
        result_path: local directory to save CSV result files
        wandb_run: active wandb run object (or None to skip W&B logging)
        target_model: optional target model; if provided and face_dataset=True,
            computes target-model feature distance as well
        robust_evaluation_models: optional list of (robust_model, eps) pairs, where each
            robust_model is a classifier with a `.model.feature_extractor` attribute and eps
            is its adversarial-training epsilon (or None if unknown). They back the
            "Robust eval model accuracy / confidence" and "Robust Inception distance"
            metric groups, which mirror their vanilla counterparts but log under a
            'robust_'-prefixed name. Each model's metrics are logged with its
            `eps_metric_suffix(eps)` appended; the un-suffixed 'robust_*' metrics hold the
            attacker-side best across the models (max for accuracy/confidence, min for
            distance). A model with eps=None gets suffix '_robust{i}' if there are several
            models, so they don't collide.
        rtpt: optional RTPT instance
        overwrite_run: if True, `wandb_run` is an existing run being updated in place;
            any summary metrics that already had a value are reported at the end
        metric_groups: iterable of metric group names (see `METRIC_GROUPS`) to actually
            compute; skipped groups leave their outputs as None. None means compute all.
        keep_existing_metrics: if True (with an existing `wandb_run`), groups whose summary
            keys all already have values are skipped entirely, and any individual existing
            summary value is never overwritten.
        cpu_optimize: if True, avoid CPU-bound overhead that doesn't affect the result:
            - the training dataset backing FID/PRDC, Inception distance and Robust
              Inception distance is built once and reused across those groups/models
              (instead of re-parsing dataset metadata -- e.g. CelebA's several
              ~200K-row CSVs -- once per group/robust model)
            - DistanceEvaluation.compute_dist, ClassificationAccuracy.compute_acc
              (Eval/Robust eval model accuracy/confidence) and
              TargetModelConfidence.compute_conf all load images with worker
              processes + pinned memory instead of serially on the main process
            Default False reproduces the exact prior behavior.
    """

    metric_groups = set(METRIC_GROUPS) if metric_groups is None else set(metric_groups)

    # Only used when cpu_optimize=True; cache is local to this call so it can't leak
    # a stale dataset into a later evaluate_attack call for a different run/config.
    _training_dataset_cache = {}

    def get_training_dataset(resize_size):
        """Build the eval training dataset (no augmentation, resized to
        `resize_size`x`resize_size`) -- shared, when cpu_optimize, across every metric
        group/robust model that uses the same resize_size, since they're otherwise
        identical datasets. Always rebuilds when cpu_optimize=False, matching the
        prior per-callsite behavior exactly."""
        if cpu_optimize and resize_size in _training_dataset_cache:
            return _training_dataset_cache[resize_size]
        config_copy = copy.deepcopy(target_config)
        config_copy.dataset.augment_data = False
        transform = get_transforms(
            config_copy, [T.Resize((resize_size, resize_size), antialias=True)], train=False)
        dataset, _, _ = get_datasets(target_config, train_transform=transform, test_transform=None)
        if cpu_optimize:
            _training_dataset_cache[resize_size] = dataset
        return dataset

    robust_specs = list(robust_evaluation_models or [])
    robust_suffixes = []
    for i, (_, eps) in enumerate(robust_specs):
        if eps is not None:
            robust_suffixes.append(eps_metric_suffix(eps))
        else:
            robust_suffixes.append(f"_robust{i}" if len(robust_specs) > 1 else "")

    if keep_existing_metrics and wandb_run:
        # Drop groups whose summary keys are all already present, so they aren't even computed.
        robust_acc_conf_keys = ['robust_correct_avg_conf', 'robust_total_avg_conf',
                                'robust_evaluation_acc@1', 'robust_evaluation_acc@5']
        robust_dist_keys = ['robust_avg_dist_evaluation']
        for s in robust_suffixes:
            robust_acc_conf_keys += [f'{k}{s}' for k in
                                     ('robust_correct_avg_conf', 'robust_total_avg_conf',
                                      'robust_evaluation_acc@1', 'robust_evaluation_acc@5')]
            robust_dist_keys.append(f'robust_avg_dist_evaluation{s}')
        group_keys = {
            "Eval model accuracy / confidence":
                ['correct_avg_conf', 'total_avg_conf', 'evaluation_acc@1', 'evaluation_acc@5'],
            "Robust eval model accuracy / confidence": robust_acc_conf_keys,
            "Target model confidence": ['target_model_avg_conf'],
            "Facenet distance": ['avg_dist_facenet'],
            "Inception distance": ['avg_dist_evaluation'],
            "Robust Inception distance": robust_dist_keys,
            "FID": ['fid_score'],
            "PRDC": ['precision', 'recall', 'density', 'coverage'],
        }
        for group, keys in group_keys.items():
            if group in metric_groups and all(wandb_run.summary.get(k, None) is not None for k in keys):
                print(f"Skipping '{group}': already computed in this run.")
                metric_groups.discard(group)
        if not metric_groups:
            print("All selected metric groups already exist in this run; nothing to do.")
            wandb.finish()
            return {}

    target_classes = torch.unique(targets).cpu().tolist()

    ####################################
    #         Attack Accuracy          #
    ####################################

    evaluation_model = torch.nn.DataParallel(evaluation_model, device_ids=gpu_devices)
    evaluation_model.to(device)
    evaluation_model.eval()

    robust_models = []
    for robust_model, _ in robust_specs:
        robust_model = torch.nn.DataParallel(robust_model, device_ids=gpu_devices)
        robust_model.to(device)
        robust_model.eval()
        robust_models.append(robust_model)

    # Per-robust-model results, one dict per model (parallel to robust_suffixes)
    robust_results = [{} for _ in robust_models]

    acc_top1 = acc_top5 = avg_eval_model_correct_conf = avg_eval_model_total_conf = None
    predictions = eval_model_confidences = maximum_eval_model_confidences = None

    if "Eval model accuracy / confidence" in metric_groups:
        class_acc_evaluator = ClassificationAccuracy(evaluation_model, device=device)

        acc_top1, acc_top5, predictions, avg_eval_model_correct_conf, avg_eval_model_total_conf, \
            eval_model_confidences, maximum_eval_model_confidences, precision_list = \
            class_acc_evaluator.compute_acc(
                attack_img_dataset,
                targets,
                batch_size=batch_size * 2,
                rtpt=rtpt, cpu_optimize=cpu_optimize)

        os.makedirs(f'{result_path}/precision_list', exist_ok=True)
        filename_precision = write_precision_list(
            f'{result_path}/precision_list/{run_id}', precision_list)
        if wandb_run:
            wandb.save(filename_precision)

        print(
            f'Evaluation of {len(attack_img_dataset)} images: '
            f'acc@1={acc_top1:.4f}, acc@5={acc_top5:.4f}, '
            f'correct_conf={avg_eval_model_correct_conf:.4f}, total_conf={avg_eval_model_total_conf:.4f}'
        )

    if "Robust eval model accuracy / confidence" in metric_groups:
        if not robust_models:
            print("Skipping 'Robust eval model accuracy / confidence': "
                  "no robust_evaluation_models provided.")
        for robust_model, suffix, results in zip(robust_models, robust_suffixes, robust_results):
            robust_class_acc_evaluator = ClassificationAccuracy(robust_model, device=device)

            robust_acc_top1, robust_acc_top5, robust_predictions, robust_correct_conf, \
                robust_total_conf, robust_eval_model_confidences, \
                robust_maximum_eval_model_confidences, robust_precision_list = \
                robust_class_acc_evaluator.compute_acc(
                    attack_img_dataset,
                    targets,
                    batch_size=batch_size * 2,
                    rtpt=rtpt, cpu_optimize=cpu_optimize)
            results.update({
                'acc@1': robust_acc_top1,
                'acc@5': robust_acc_top5,
                'correct_conf': robust_correct_conf,
                'total_conf': robust_total_conf,
            })

            os.makedirs(f'{result_path}/precision_list', exist_ok=True)
            filename_robust_precision = write_precision_list(
                f'{result_path}/precision_list/robust{suffix}_{run_id}', robust_precision_list)
            if wandb_run:
                wandb.save(filename_robust_precision)

            print(
                f'Robust evaluation{suffix} of {len(attack_img_dataset)} images: '
                f'acc@1={robust_acc_top1:.4f}, acc@5={robust_acc_top5:.4f}, '
                f'correct_conf={robust_correct_conf:.4f}, '
                f'total_conf={robust_total_conf:.4f}'
            )

    ####################################
    #  Target Model Confidences        #
    ####################################

    target_model_avg_conf = None
    target_model_conf_by_class = None
    target_model_confidences = None

    if target_model is not None and "Target model confidence" in metric_groups:
        try:
            target_attack_dataset = copy.copy(attack_img_dataset)
            target_attack_dataset.transform = get_transforms(target_config, train=False)

            target_model_wrapped = torch.nn.DataParallel(target_model, device_ids=gpu_devices)
            target_model_wrapped.to(device)
            conf_evaluator = TargetModelConfidence(target_model_wrapped, device=device)
            target_model_avg_conf, target_model_conf_by_class_list, target_model_confidences = conf_evaluator.compute_conf(
                target_attack_dataset, targets, batch_size=batch_size * 2, cpu_optimize=cpu_optimize)

            print(f'Target model avg conf on reconstructions: {target_model_avg_conf:.4f}')

            os.makedirs(f'{result_path}/target_model_conf_by_class', exist_ok=True)
            filename_target_conf = write_precision_list(
                f'{result_path}/target_model_conf_by_class/{run_id}',
                target_model_conf_by_class_list)
            if wandb_run:
                wandb.save(filename_target_conf)

            target_model_conf_by_class = {
                row[0]: row[1] for row in target_model_conf_by_class_list[1:]}

        except Exception:
            print(traceback.format_exc())

    ####################################
    #    FID Score and GAN Metrics     #
    ####################################

    fid_score = None
    fid_by_class = None
    precision, recall = None, None
    density, coverage = None, None

    if metric_groups & {"FID", "PRDC"}:
        try:
            training_dataset = get_training_dataset(299)
            training_subset = ClassSubset(training_dataset, target_classes=target_classes)

            fid_evaluation = FID_Score(
                training_subset,
                attack_img_dataset,
                device=device,
                batch_size=batch_size * 3,
                dims=2048,
                num_workers=0,
                gpu_devices=gpu_devices)

            print("Computing FID score...")
            fid_score = fid_evaluation.compute_fid(rtpt)
            print(f'FID score: {fid_score:.4f}')

            print("Computing per-class FID scores...")
            fid_by_class = fid_evaluation.compute_fid_by_class()
            print(f'Per-class FID: { {k: f"{v:.4f}" for k, v in fid_by_class.items()} }')
            os.makedirs(f'{result_path}/fid_by_class', exist_ok=True)
            fid_by_class_list = [['target', 'fid_score']] + \
                [[cls, val] for cls, val in fid_by_class.items()]
            filename_fid_by_class = write_precision_list(
                f'{result_path}/fid_by_class/{run_id}', fid_by_class_list)
            if wandb_run:
                wandb.save(filename_fid_by_class)

            print("Computing PRDC metrics...")
            prdc = PRDC(
                training_subset,
                attack_img_dataset,
                device=device,
                batch_size=batch_size * 3,
                dims=2048,
                num_workers=0,
                gpu_devices=gpu_devices)
            precision, recall, density, coverage = prdc.compute_metric(
                target_classes=target_classes, k=3, rtpt=rtpt)
            if precision is not None:
                print(f'Precision={precision:.4f}, Recall={recall:.4f}, '
                      f'Density={density:.4f}, Coverage={coverage:.4f}')

            del training_dataset, training_subset

        except Exception:
            print(traceback.format_exc())

    ####################################
    #         Feature Distance         #
    ####################################

    avg_dist_inception = None
    avg_dist_facenet = None
    mean_distances_list_inception = None
    mean_distances_list_facenet = None
    evaluation_model_dist = None

    if "Inception distance" in metric_groups:
        try:
            if isinstance(evaluation_model, torch.nn.DataParallel):
                evaluation_model_dist = evaluation_model.module.model.feature_extractor
            else:
                evaluation_model_dist = evaluation_model.model.feature_extractor
            evaluation_model_dist.to(device)
            evaluation_model_dist.eval()

            training_dataset_inception = get_training_dataset(299)

            evaluate_inception = DistanceEvaluation(
                evaluation_model_dist, 299, training_dataset_inception)
            avg_dist_inception, mean_distances_list_inception = evaluate_inception.compute_dist(
                attack_img_dataset, targets,
                batch_size=batch_size_single * 5, rtpt=rtpt, cpu_optimize=cpu_optimize)

            os.makedirs(f'{result_path}/distance_inceptionv3_list', exist_ok=True)
            filename_distance = write_precision_list(
                f'{result_path}/distance_inceptionv3_list/{run_id}',
                mean_distances_list_inception)
            if wandb_run:
                wandb.save(filename_distance)
            print('Mean Distance on Inception-v3:', avg_dist_inception.cpu().item())

            del training_dataset_inception

        except Exception:
            print(traceback.format_exc())

    if "Robust Inception distance" in metric_groups:
        if not robust_models:
            print("Skipping 'Robust Inception distance': no robust_evaluation_models provided.")
        for robust_model, suffix, results in zip(robust_models, robust_suffixes, robust_results):
            try:
                if isinstance(robust_model, torch.nn.DataParallel):
                    robust_evaluation_model_dist = robust_model.module.model.feature_extractor
                else:
                    robust_evaluation_model_dist = robust_model.model.feature_extractor
                robust_evaluation_model_dist.to(device)
                robust_evaluation_model_dist.eval()

                robust_training_dataset_inception = get_training_dataset(299)

                robust_evaluate_inception = DistanceEvaluation(
                    robust_evaluation_model_dist, 299, robust_training_dataset_inception)
                robust_avg_dist_inception, robust_mean_distances_list_inception = \
                    robust_evaluate_inception.compute_dist(
                        attack_img_dataset, targets,
                        batch_size=batch_size_single * 5, rtpt=rtpt, cpu_optimize=cpu_optimize)

                os.makedirs(f'{result_path}/distance_inceptionv3_list', exist_ok=True)
                results['dist'] = robust_avg_dist_inception.cpu().item()
                filename_robust_distance = write_precision_list(
                    f'{result_path}/distance_inceptionv3_list/robust{suffix}_{run_id}',
                    robust_mean_distances_list_inception)
                if wandb_run:
                    wandb.save(filename_robust_distance)
                print(f'Robust mean distance on Inception-v3{suffix}:', results['dist'])

                del robust_training_dataset_inception

            except Exception:
                print(traceback.format_exc())

    if "Facenet distance" in metric_groups and target_config.dataset.face_dataset:
        try:
            facenet = InceptionResnetV1(pretrained='vggface2')
            facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
            facenet.to(device)
            facenet.eval()

            face_net_img_size = 160
            training_dataset_facenet = get_training_dataset(face_net_img_size)

            evaluater_facenet = DistanceEvaluation(
                facenet, face_net_img_size, training_dataset_facenet)
            avg_dist_facenet, mean_distances_list_facenet = evaluater_facenet.compute_dist(
                attack_img_dataset, targets,
                batch_size=batch_size_single * 8,
                resize=face_net_img_size,
                rtpt=rtpt, cpu_optimize=cpu_optimize)

            os.makedirs(f'{result_path}/distance_facenet_list', exist_ok=True)
            filename_distance = write_precision_list(
                f'{result_path}/distance_facenet_list/{run_id}',
                mean_distances_list_facenet)
            if wandb_run:
                wandb.save(filename_distance)
            print('Mean Distance on FaceNet:', avg_dist_facenet.cpu().item())

            del training_dataset_facenet

        except Exception:
            print(traceback.format_exc())

    ####################################
    #          Finish Logging          #
    ####################################

    if rtpt:
        rtpt.step(subtitle='Finishing up')

    print('Finishing evaluation, logging results and sample images.')

    if wandb_run and "sample image logging (v1)" in metric_groups:
        # Sample images from first and last N/2 classes for wandb logging
        num_imgs = 20
        label_subset = select_log_classes(targets)

        log_imgs = []
        log_targets = []
        log_predictions = []
        log_eval_model_max_confidences = []
        log_eval_model_confidences = []

        pred_tensor = torch.tensor(predictions) if predictions is not None else None
        max_eval_model_conf_tensor = \
            torch.tensor(maximum_eval_model_confidences) if maximum_eval_model_confidences is not None else None
        eval_model_conf_tensor = \
            torch.tensor(eval_model_confidences) if eval_model_confidences is not None else None

        for label in label_subset:
            mask = targets == label
            indices = mask.nonzero(as_tuple=False).squeeze(1)[:num_imgs].tolist()
            imgs_for_label = torch.stack([attack_img_dataset[i][0] for i in indices])
            flat_mask = torch.zeros(len(targets), dtype=torch.bool)
            flat_mask[indices] = True

            log_imgs.append(imgs_for_label)
            log_targets += [label] * len(indices)
            if pred_tensor is not None:
                log_predictions.append(pred_tensor[flat_mask][:num_imgs])
            if max_eval_model_conf_tensor is not None:
                log_eval_model_max_confidences.append(max_eval_model_conf_tensor[flat_mask][:num_imgs])
            if eval_model_conf_tensor is not None:
                log_eval_model_confidences.append(eval_model_conf_tensor[flat_mask][:num_imgs])

        log_imgs = torch.cat(log_imgs, dim=0)
        log_predictions = torch.cat(log_predictions, dim=0) if log_predictions else None
        log_eval_model_max_confidences = \
            torch.cat(log_eval_model_max_confidences, dim=0) if log_eval_model_max_confidences else None
        log_eval_model_confidences = \
            torch.cat(log_eval_model_confidences, dim=0) if log_eval_model_confidences else None

        if log_predictions is not None and log_eval_model_max_confidences is not None \
                and log_eval_model_confidences is not None:
            log_final_images(log_imgs, log_predictions, log_eval_model_max_confidences,
                             log_eval_model_confidences, idx_to_class)

        if evaluation_model_dist is not None:
            log_nearest_neighbors(log_imgs, log_targets,
                                  evaluation_model_dist, 'InceptionV3',
                                  target_dataset, img_size=299,
                                  seed=target_config.training.seed)

        if target_config.dataset.face_dataset:
            facenet_log = InceptionResnetV1(pretrained='vggface2')
            facenet_log.to(device)
            facenet_log.eval()
            log_nearest_neighbors(log_imgs, log_targets,
                                  facenet_log, 'FaceNet',
                                  target_dataset, img_size=160,
                                  seed=target_config.training.seed)

        # Distance histogram (Inception)
        if mean_distances_list_inception is not None:
            distances_arr = np.array(
                [row[1] for row in mean_distances_list_inception[1:]])
            plt_inc = _plot_histogram(
                distances_arr, 20,
                'Distribution of Feature Distances (Inception)',
                'Feature Distance', 'Frequency', color='lightcoral')
            wandb.log({'Distances Histogram (Inception)': wandb.Image(plt_inc)})
            plt.close()

        # Distance histogram (FaceNet)
        if mean_distances_list_facenet is not None:
            distances_arr = np.array(
                [row[1] for row in mean_distances_list_facenet[1:]])
            plt_fn = _plot_histogram(
                distances_arr, 20,
                'Distribution of Feature Distances (FaceNet)',
                'Feature Distance', 'Frequency', color='skyblue')
            wandb.log({'Distances Histogram (FaceNet)': wandb.Image(plt_fn)})
            plt.close()

        # Confidence histogram
        if eval_model_confidences is not None:
            plt_conf = _plot_histogram(
                eval_model_confidences, 30,
                'Distribution of Target Confidences',
                'Confidence', 'Frequency', color='lightgreen')
            wandb.log({'Confidence Histogram': wandb.Image(plt_conf)})
            plt.close()

    if wandb_run and "sample image logging (v2)" in metric_groups:
        if predictions is None or target_model_confidences is None:
            print("Skipping sample image logging (v2): requires both "
                  "'Eval model accuracy / confidence' and 'Target model confidence' to be enabled.")
        else:
            for class_label in select_log_classes(targets):
                columns = build_percentile_columns(
                    attack_img_dataset, targets, eval_model_confidences,
                    target_model_confidences, class_label)
                num_imgs_per_col = columns['num_imgs_per_col']
                class_name = idx_to_class[class_label] if idx_to_class is not None else class_label

                ref_indices = (target_dataset.targets == class_label).nonzero(as_tuple=False)
                ref_img = None
                if len(ref_indices) > 0:
                    ref_img = (target_dataset[ref_indices[0].item()][0].clamp(-1, 1) + 1) / 2

                nrows = num_imgs_per_col + 1
                fig = plt.figure(figsize=(6, 2.2 * nrows))
                gs = fig.add_gridspec(nrows, 2)
                render_reference_row(
                    [fig.add_subplot(gs[0, :])],
                    [ref_img] if ref_img is not None else [],
                    label='Real example')
                render_percentile_column(
                    [fig.add_subplot(gs[row, 0]) for row in range(1, nrows)],
                    columns['left'], columns['percentiles'],
                    header='Eval model confidence')
                render_percentile_column(
                    [fig.add_subplot(gs[row, 1]) for row in range(1, nrows)],
                    columns['right'], columns['percentiles'],
                    header='Target model confidence')
                fig.suptitle(f'Class {class_name}')
                fig.tight_layout()
                wandb.log({f'Sample Image Logging (v2)/class_{class_label}': wandb.Image(fig)})
                plt.close(fig)

                class_dir = f'model_inversion/results/sample_image_v2/{run_id}/class_{class_label}'
                os.makedirs(class_dir, exist_ok=True)
                v2_path = f'{class_dir}/data.pt'
                torch.save(columns, v2_path)
                wandb.save(v2_path)

    if wandb_run:
        # Summary metrics -- built up one metric group at a time so an unchecked group's
        # keys never even enter the dict, rather than relying on their value being None.
        summary_metrics = {}

        if "Eval model accuracy / confidence" in metric_groups:
            summary_metrics.update({
                'correct_avg_conf': avg_eval_model_correct_conf,
                'total_avg_conf': avg_eval_model_total_conf,
                'evaluation_acc@1': acc_top1,
                'evaluation_acc@5': acc_top5,
            })

        def best_across_robust(result_key, best_fn):
            """Attacker-side best of a metric across the robust models that produced it."""
            values = [r[result_key] for r in robust_results if r.get(result_key) is not None]
            return best_fn(values) if values else None

        if "Robust eval model accuracy / confidence" in metric_groups:
            # (result key, summary name, best-of aggregate): higher is better for the attacker
            for result_key, name in [('correct_conf', 'robust_correct_avg_conf'),
                                     ('total_conf', 'robust_total_avg_conf'),
                                     ('acc@1', 'robust_evaluation_acc@1'),
                                     ('acc@5', 'robust_evaluation_acc@5')]:
                for suffix, results in zip(robust_suffixes, robust_results):
                    summary_metrics[f'{name}{suffix}'] = results.get(result_key)
                # un-suffixed = best across all robust eval models (max)
                summary_metrics[name] = best_across_robust(result_key, max)

        if "Target model confidence" in metric_groups:
            summary_metrics['target_model_avg_conf'] = target_model_avg_conf

        if "Facenet distance" in metric_groups:
            summary_metrics['avg_dist_facenet'] = \
                avg_dist_facenet.cpu().item() if avg_dist_facenet is not None else None

        if "Inception distance" in metric_groups:
            summary_metrics['avg_dist_evaluation'] = \
                avg_dist_inception.cpu().item() if avg_dist_inception is not None else None

        if "Robust Inception distance" in metric_groups:
            for suffix, results in zip(robust_suffixes, robust_results):
                summary_metrics[f'robust_avg_dist_evaluation{suffix}'] = results.get('dist')
            # un-suffixed = best across all robust eval models (min: closest to real images)
            summary_metrics['robust_avg_dist_evaluation'] = best_across_robust('dist', min)

        if "FID" in metric_groups:
            summary_metrics['fid_score'] = fid_score

        if "PRDC" in metric_groups:
            summary_metrics.update({
                'precision': precision,
                'recall': recall,
                'density': density,
                'coverage': coverage,
            })

        # Even within a checked group, a value can still end up None (e.g. an exception
        # was caught during its computation) -- skip logging those rather than deleting
        # a prior good value with None.
        metrics_to_write = {}
        overwritten_metrics = []
        for key, value in summary_metrics.items():
            if value is not None:
                if keep_existing_metrics and wandb_run.summary.get(key, None) is not None:
                    continue
                if overwrite_run:
                    old_value = wandb_run.summary.get(key, None)
                    if old_value is not None:
                        overwritten_metrics.append((key, old_value, value))
                metrics_to_write[key] = value

        if overwrite_run and overwritten_metrics:
            print(f"\nThe following {len(overwritten_metrics)} existing W&B summary "
                  f"metric(s) on run {wandb_run.name} ({wandb_run.id}) will be overwritten:")
            for key, old_value, new_value in overwritten_metrics:
                print(f'  {key}: {old_value} -> {new_value}')
            confirm = input("Proceed with overwriting these? [y/N]: ").strip().lower()
            if confirm not in ('y', 'yes'):
                print("Skipping overwrite of existing metrics for this run "
                      "(any genuinely new metrics below will still be logged).")
                overwritten_keys = {key for key, _, _ in overwritten_metrics}
                metrics_to_write = {k: v for k, v in metrics_to_write.items()
                                    if k not in overwritten_keys}
                overwritten_metrics = []

        for key, value in metrics_to_write.items():
            wandb.run.summary[key] = value

        if fid_by_class and "FID" in metric_groups:
            wandb.log({'fid_by_class': fid_by_class})

        if overwrite_run and overwritten_metrics:
            print('\nOverwritten W&B summary metrics:')
            for key, old_value, new_value in overwritten_metrics:
                print(f'  {key}: {old_value} -> {new_value}')

        wandb.finish()

    return {
        'acc_top1': acc_top1,
        'acc_top5': acc_top5,
        'avg_eval_model_correct_conf': avg_eval_model_correct_conf,
        'avg_eval_model_total_conf': avg_eval_model_total_conf,
        'target_model_avg_conf': target_model_avg_conf,
        'target_model_conf_by_class': target_model_conf_by_class,
        'fid_score': fid_score,
        'fid_by_class': fid_by_class,
        'precision': precision,
        'recall': recall,
        'density': density,
        'coverage': coverage,
        'avg_dist_inception': avg_dist_inception,
        'avg_dist_facenet': avg_dist_facenet,
    }


def _plot_histogram(data, bins, title, xlabel, ylabel, color):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=bins, edgecolor='black', color=color, alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.8)
    fig.tight_layout()
    return fig
