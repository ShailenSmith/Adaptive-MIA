
#! WIP -- don't push to public yet
import torch
import wandb
import json
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import shutil
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torchvision import datasets, transforms
import torchvision

from copy import deepcopy
from torchvision import transforms as T

import sys
sys.path.append(os.getcwd())

from pick import pick

from classifiers.get_model import get_model
from defenses.get_defense import get_defense
from data_processing.data_augmentation import get_transforms
from data_processing.datasets import get_datasets
from model_inversion.plug_and_play.ppa_evaluate import ppa_evaluate
from utils.wandb_helpers import get_config, get_weights
from utils.load_trained_models import get_target_config_and_weights, get_evaluation_config_and_weights
from Plug_and_Play_Attacks.utils.stylegan import load_generator
import model_inversion.plug_and_play.modify_to_ppa_repo as ppa_modify # import model_compatibility_wrapper, convert_configs

from Plug_and_Play_Attacks.utils.attack_config_parser import AttackConfigParser
from Plug_and_Play_Attacks.utils.datasets import (create_target_dataset, get_facescrub_idx_to_class,
                            get_stanford_dogs_idx_to_class)

from Plug_and_Play_Attacks.utils.stylegan import create_image

from model_inversion.model_inversion_evaluate import evaluate_attack, METRIC_GROUPS
from experiments.all_run_ids import ALL_RUN_IDS

# ModelInversion wandb run ID for the robust evaluation model backing the "robust eval
# model accuracy / confidence" and "Robust Inception distance" metric groups.
DATASET_TO_ROBUST_EVAL_WANDB_ID = {
    "FaceScrub": "4bc7jytx",
    "CelebA-eps.0025": "ph9f9uc4",
    "CelebA-eps.031": "tt13g4pu",
}

EVAL_WANDB_ID_TO_EPS = {
    "4bc7jytx": 0.031,
    "ph9f9uc4": 0.0025,
    "tt13g4pu": 0.031,
}

ROBUST_METRIC_GROUPS = {"Robust eval model accuracy / confidence", "Robust Inception distance"}

# Only used when cpu_optimize=True, for the thread pool that writes generated attack
# images to disk. This is I/O-bound (PNG encode + file write), not CPU-core-bound like
# DataLoader workers, so a somewhat higher worker count than
# mi_distance_metrics.CPU_OPTIMIZE_NUM_WORKERS is reasonable here.
IMAGE_SAVE_NUM_WORKERS = min(8, (os.cpu_count() or 1) * 2)


def pick_metric_groups():
    title = "Choose metric groups to compute (space to toggle, enter to confirm):"
    selected = pick(METRIC_GROUPS, title, multiselect=True, min_selection_count=1)
    return {label for label, _ in selected}


def load_robust_evaluation_model(entity, dataset):
    """Load the robust evaluation model backing the 'Robust eval model accuracy /
    confidence' and 'Robust Inception distance' metric groups, from the hardcoded
    ROBUST_EVAL_WANDB_ID run in the ModelInversion project. Returns
    (robust_evaluation_model, robust_eval_run_id) -- the run ID is used to look up
    the model's AT epsilon in EVAL_WANDB_ID_TO_EPS for metric naming."""
    if not DATASET_TO_ROBUST_EVAL_WANDB_ID:
        raise ValueError(
            "DATASET_TO_ROBUST_EVAL_WANDB_ID is not set -- fill in a ModelInversion wandb run ID "
            "for the robust evaluation model at the top of run_wandb_evaluate.py.")

    robust_eval_run_id = DATASET_TO_ROBUST_EVAL_WANDB_ID.get(dataset, None)
    if not robust_eval_run_id:
        raise ValueError(f"No robust evaluation model found for dataset {dataset}.")

    robust_config, _ = get_config(entity, "ModelInversion", robust_eval_run_id)
    robust_weights_path = get_weights(
        entity=entity, project="ModelInversion", run_id=robust_eval_run_id,
        save_as=robust_config.training.save_as)

    robust_evaluation_model = get_model(robust_config)
    robust_evaluation_model.load_model(robust_weights_path)
    robust_evaluation_model = ppa_modify.model_compatibility_wrapper(
        model=robust_evaluation_model, target_config=robust_config)
    return robust_evaluation_model, robust_eval_run_id


def pick_robust_evaluation_models(entity):
    """Pick one or more robust evaluation models (by dataset key) and load them. Returns
    a list of (robust_evaluation_model, eps) pairs for evaluate_attack."""
    title = "Choose robust evaluation model(s) (space to toggle, enter to confirm):"
    selected = pick(list(DATASET_TO_ROBUST_EVAL_WANDB_ID.keys()), title,
                    multiselect=True, min_selection_count=1)
    robust_evaluation_models = []
    for dataset, _ in selected:
        model, robust_eval_run_id = load_robust_evaluation_model(entity, dataset)
        robust_evaluation_models.append((model, EVAL_WANDB_ID_TO_EPS.get(robust_eval_run_id)))
    return robust_evaluation_models


def pick_target_wandb_ids():
    """Pick target model(s) via the same group -> {model_name: target_wandb_id} structure
    used in scripts/combine_wandb_runs.py. Returns {model_name: target_wandb_id}."""
    title = "Select which run ID groups to include"
    selected_groups = pick(list(ALL_RUN_IDS.keys()), title, multiselect=True, min_selection_count=1)
    selected_group_names = [option for option, _ in selected_groups]

    pooled_run_ids = {}
    for group_name in selected_group_names:
        pooled_run_ids.update(ALL_RUN_IDS[group_name])

    title = "Select target model(s) to evaluate attacks against"
    run_options = list(pooled_run_ids.keys()) + ["All models"]
    selected_models = pick(run_options, title, multiselect=True, min_selection_count=1)
    selected_model_names = [option for option, _ in selected_models]

    if "All models" in selected_model_names:
        return pooled_run_ids
    return {name: pooled_run_ids[name] for name in selected_model_names}


def find_matching_runs(api, entity, project, target_wandb_id):
    """Find runs in `project` whose config.target_wandb_id (or load_from_wandb_id, for
    older runs) matches `target_wandb_id`."""
    matches = []
    for run in api.runs(f"{entity}/{project}"):
        current_target_wandb_id = run.config.get("target_wandb_id", None)
        if current_target_wandb_id is None:
            current_target_wandb_id = run.config.get("load_from_wandb_id", None)
        if current_target_wandb_id == target_wandb_id:
            matches.append(run)
    return matches


def _lookup_config(run, dotted_key):
    """Look up a dotted key (e.g. 'attack.attack.align_method') in run.config, trying
    the literal flat key first and then walking nested dicts. Missing -> None."""
    if dotted_key in run.config:
        return run.config[dotted_key]
    value = run.config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _matches_any(value, allowed):
    """True if `value` equals any entry of `allowed` (None entries match missing/null;
    other entries compare as strings so 100 == "100")."""
    if value is None:
        return None in allowed
    return str(value) in {str(a) for a in allowed if a is not None}


def _is_universal_0921(run):
    """universal_0921 == True, checked in config, then summary, then tags."""
    if _lookup_config(run, "universal_0921") is True:
        return True
    try:
        if bool(run.summary.get("universal_0921", None)) is True:
            return True
    except Exception:
        pass
    return "universal_0921" in (run.tags or [])


# Mirrors the W&B UI filters used to browse PPA-main runs (only the checked ones).
PPA_MAIN_FILTERS = [
    ("universal_0921 == True", _is_universal_0921),
    ("align_method in (old, vanilla, null)",
     lambda run: _matches_any(_lookup_config(run, "attack.attack.align_method"),
                              ["old", "vanilla", None])),
    ("transformations.RandomErasing.p is null",
     lambda run: _lookup_config(run, "attack.attack.transformations.RandomErasing.p") is None),
    # ("attack_loss_function == negative_label_smoothing",
    #  lambda run: _lookup_config(run, "attack.attack.attack_loss_function")
    #              == "negative_label_smoothing"),
    ("targets in (100, all, 75)",
     lambda run: _matches_any(_lookup_config(run, "attack.attack.targets"), [100, "all", 75])),
]


def apply_ppa_main_filters(runs):
    filtered_runs = []
    for run in runs:
        if all(check(run) for _, check in PPA_MAIN_FILTERS):
            filtered_runs.append(run)
    return filtered_runs


def pick_runs_to_evaluate(entity, source_project):
    """Pick target model(s) by name, then resolve each to the attack run(s) in
    `source_project` that targeted it. Always shows a picker over the matching runs --
    even when there's only one -- so the user can opt out of evaluating a given target
    model entirely. Returns a list of wandb Run objects to evaluate."""
    model_name_to_target_id = pick_target_wandb_ids()

    use_ppa_main_filters = False
    if source_project == "PPA-main":
        title = "Apply the universal_0921 PPA-main run filters?\n" + "\n".join(
            f"  - {desc}" for desc, _ in PPA_MAIN_FILTERS)
        choice, _ = pick(["Yes", "No"], title)
        use_ppa_main_filters = choice == "Yes"

    choice, _ = pick(["Yes", "No"], "Also filter to runs missing 'robust_total_avg_conf'?")
    require_missing_robust_conf = choice == "Yes"

    api = wandb.Api()
    runs_to_evaluate = []
    for model_name, target_wandb_id in model_name_to_target_id.items():
        matches = find_matching_runs(api, entity, source_project, target_wandb_id)
        if use_ppa_main_filters:
            matches = apply_ppa_main_filters(matches)
        if require_missing_robust_conf:
            matches = [run for run in matches if run.summary.get('robust_total_avg_conf', None) is None]
        if not matches:
            print(f"WARNING: No {source_project} runs found targeting {model_name} "
                  f"(target_wandb_id={target_wandb_id}), skipping.")
            continue

        title = (f"{model_name} (target_wandb_id={target_wandb_id}): choose which "
                f"{source_project} run(s) to evaluate, or none to skip "
                f"(showing {len(matches)} run{'s' if len(matches) != 1 else ''}):")
        options = [f"{run.name} ({run.id})" + (" (STILL RUNNING)" if run.state == "running" else "")
                   for run in matches]
        selected = pick(options, title, multiselect=True, min_selection_count=0)
        selected_indexes = {index for _, index in selected}
        runs_to_evaluate += [matches[i] for i in selected_indexes]

    return runs_to_evaluate


def log_target_metadata(run, attack_config, target_config):
    """Save which target model this evaluation run's attack was targeting, plus that
    target model's dataset/architecture/defense, onto the evaluation run's config."""
    architecture = getattr(target_config.model, 'architecture', target_config.model.name)
    run.config.update({
        'target_wandb_id': attack_config.target_wandb_id,
        'dataset': target_config.dataset.dataset,
        'architecture': architecture,
        'defense': target_config.defense.name,
    }, allow_val_change=True)


def ppa_evaluate_from_wandb(entity, source_project, dest_project, run_id, evaluation_model,
                            option, new_run_id=None, new_run_name=None, manual_load={},
                            overwrite_existing_run=False, metric_groups=None,
                            robust_evaluation_models=None,
                            keep_existing_metrics=False, cpu_optimize=False):

    # Set devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]
    
    attack_config, run_name = get_config(entity, source_project, run_id)

    if "model" in attack_config:
        attack_config.attack = attack_config.model

    # set random seed to match attack_config
    torch.manual_seed(attack_config.training.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    target_config, target_weights_path = get_target_config_and_weights(attack_config)
    evaluation_config, evaluation_weights_path = get_evaluation_config_and_weights(attack_config)

    # Load target model
    target_model = get_model(target_config)
    target_model = get_defense(config=target_config, model=target_model)
    target_model.load_model(target_weights_path)
    target_model = ppa_modify.model_compatibility_wrapper(model = target_model, target_config = target_config)


    
    # Load evaluation model
    evaluation_model = get_model(evaluation_config)
    evaluation_model.load_model(evaluation_weights_path)
    evaluation_model = ppa_modify.model_compatibility_wrapper(model = evaluation_model, target_config = evaluation_config)

    new_attack_config_path = ppa_modify.convert_configs(target_config, attack_config)
    new_attack_config = AttackConfigParser(new_attack_config_path)

    api = wandb.Api()
    run = api.run(f"{entity}/{source_project}/{run_id}")

    # load targets
    actual_targets_path = run.file(f"model_inversion/plug_and_play/results/targets/{run_id}.pt").download(exist_ok=True)
    print(f"Downloaded targets from wandb: {actual_targets_path.name}")
    with open(actual_targets_path.name, 'rb') as f:
        targets = torch.load(f)
    
    # load final_targets
    actual_final_targets_path = run.file(f"model_inversion/plug_and_play/results/final_targets/{run_id}.pt").download(exist_ok=True)
    print(f"Downloaded final_targets from wandb: {actual_final_targets_path.name}")
    with open(actual_final_targets_path.name, 'rb') as f:
        final_targets = torch.load(f)

    # load w_optimized_unselected
    w_optimized_unselected_path = run.file(f"model_inversion/plug_and_play/results/optimized_w/{run_id}.pt").download(exist_ok=True)
    print(f"Downloaded optimized_unselected w from wandb: {w_optimized_unselected_path.name}")
    with open(w_optimized_unselected_path.name, 'rb') as f:
        w_optimized_unselected = torch.load(f)

    # load final_w
    final_w_path = run.file(f"model_inversion/plug_and_play/results/optimized_w_selected/{run_id}.pt").download(replace=True, exist_ok=True)
    print(f"Downloaded final w from wandb: {final_w_path.name}")
    with open(final_w_path.name, 'rb') as f:
        final_w = torch.load(f)

    # Load data
    transform = get_transforms(target_config, train=False)
    target_dataset, _, _ = get_datasets(config=target_config,
                                        train_transform=transform,
                                        test_transform=None)

    # Load idx to class mappings
    idx_to_class = None
    if new_attack_config.dataset.lower() == 'facescrub':
        idx_to_class = get_facescrub_idx_to_class()
    elif new_attack_config.dataset.lower() == 'stanford_dogs':
        idx_to_class = get_stanford_dogs_idx_to_class()
    else:
        class KeyDict(dict):
            def __missing__(self, key):
                return key
        idx_to_class = KeyDict()

    # Load generator
    G = load_generator(attack_config['attack']['stylegan_model'], device)
    G.img_channels = target_dataset[0][0].shape[0] # Extracting the number of channels from the dataset
    num_ws = G.num_ws
    synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    synthesis.num_ws = num_ws
    del G

    # batch size stuff
    batch_size_single = new_attack_config.attack['batch_size']
    batch_size = new_attack_config.attack['batch_size'] * max(torch.cuda.device_count(), 1) #! Changed this to max(n, 1) to make it work with cpu.

    # initialize wandb (assumes WANDB_API_KEY is set as an environment variable)
    if overwrite_existing_run:
        new_run = wandb.init(project=source_project,
            entity=entity,
            id=run_id,
            resume="must",
        )
    else:
        new_run = wandb.init(project=dest_project,
            entity=entity,
            id=new_run_id,
            name=new_run_name,
            resume="never",
        )
    print(f"wandb initiated with run id: {new_run.id} and run name: {new_run.name}")
    log_target_metadata(new_run, attack_config, target_config)

    # run evaluation

    ### make/load imgfolder dataset ###
    attack_img_dataset_path = os.path.join("model_inversion", "plug_and_play", "results", "attack_img_datasets", run_id)

    # check if directory is empty
    if not os.path.isdir(attack_img_dataset_path) or len(os.listdir(attack_img_dataset_path)) < 5:
        if os.path.isdir(attack_img_dataset_path):
            shutil.rmtree(attack_img_dataset_path)
        os.makedirs(attack_img_dataset_path)

        # set up folder names for dataset
        max_string_length = len(str(final_targets.max().item())) + 1
        folder_names = {}
        for i in final_targets.unique():
            i = i.item()
            string_length = len(str(i))
            if string_length < max_string_length:
                folder_name = str(i).zfill(max_string_length)
            else:
                folder_name = str(i)

            os.makedirs(os.path.join(attack_img_dataset_path, folder_name), exist_ok=True)
            folder_names[i] = folder_name

        # make attack images
        # cpu_optimize: PNG encode + disk write is real CPU work that otherwise fully
        # serializes with the next batch's GPU generation (the loop can't start
        # generating batch N+1 until every image in batch N has been saved). Farming
        # the saves out to worker threads lets them run while the GPU keeps going, and
        # moving the whole batch to CPU at once instead of image-by-image cuts down on
        # sync points. Doesn't change any saved image; default False saves exactly as
        # before, one image at a time on the main thread.
        save_executor = ThreadPoolExecutor(max_workers=IMAGE_SAVE_NUM_WORKERS) if cpu_optimize else None
        pending_saves = []
        attack_dataset = TensorDataset(final_w, final_targets)
        for batch_idx, (w_batch, target_batch) in tqdm(enumerate(
                            DataLoader(attack_dataset, batch_size=batch_size, shuffle=False)),
                            desc="Creating attack images", total=len(attack_dataset)//batch_size):
            w_batch, target_batch = w_batch.to(device), target_batch.to(device)

            img_batch = create_image(w_batch,
                                synthesis,
                                new_attack_config.attack_center_crop,
                                resize=299,
                                batch_size=batch_size)

            if cpu_optimize:
                img_batch_cpu = (img_batch.detach().cpu() + 1) / 2  # un-normalize the whole batch at once
                for i in range(img_batch_cpu.shape[0]):
                    label = target_batch[i].item()
                    folder_name = folder_names[label]
                    save_path = os.path.join(attack_img_dataset_path, folder_name, f"{batch_idx}_{i}.png")
                    pending_saves.append(save_executor.submit(
                        torchvision.utils.save_image, img_batch_cpu[i], save_path))
            else:
                for i in range(img_batch.shape[0]):
                    img = img_batch[i].detach().cpu()
                    label = target_batch[i].item()

                    folder_name = folder_names[label]

                    save_path = os.path.join(attack_img_dataset_path, folder_name, f"{batch_idx}_{i}.png")
                    img = (img + 1) / 2 # un-normalize the image
                    torchvision.utils.save_image(img, save_path)

        if cpu_optimize:
            # Every image must be on disk before the ImageFolder below scans this
            # directory -- wait for (and propagate errors from) all pending saves.
            for future in pending_saves:
                future.result()
            save_executor.shutdown()
    else:
        print(f"Attack image dataset already exists at {attack_img_dataset_path}, skipping dataset generation.")

    if option == "Dataset generation":
        print("Dataset generation complete. Exiting.")
        return
    
    else: # continue with evaluation
        old_augmentations = False
        if old_augmentations:
            no_augmentations = get_transforms(target_config, train=False)
            attack_img_dataset = datasets.ImageFolder(attack_img_dataset_path, transform=no_augmentations) #, allow_empty=True)
        else:
            eval_transform = get_transforms(evaluation_config, train=False)
            attack_img_dataset = datasets.ImageFolder(attack_img_dataset_path, transform=eval_transform)

        # Map ImageFolder-internal indices → original class IDs so the DataLoader
        # labels and the targets tensor are both in original-class-ID space.
        idx_to_original = {v: int(k) for k, v in attack_img_dataset.class_to_idx.items()}
        attack_img_dataset.targets = [idx_to_original[t] for t in attack_img_dataset.targets]
        attack_img_dataset.target_transform = idx_to_original.__getitem__
        final_targets = torch.tensor(attack_img_dataset.targets)

        result_path = os.path.join("model_inversion", "results", f"{new_run.name}_{new_run.id}")
        os.makedirs(result_path, exist_ok=True)

        evaluate_attack(
            evaluation_model=evaluation_model,
            # config=new_attack_config,
            target_config=target_config,
            attack_img_dataset=attack_img_dataset,
            targets=final_targets,
            device=device,
            gpu_devices=gpu_devices,
            idx_to_class=idx_to_class,
            batch_size=batch_size,
            batch_size_single=batch_size_single,
            run_id=new_run.id,
            target_dataset=target_dataset,
            result_path=result_path,
            wandb_run=new_run,
            target_model=target_model,
            robust_evaluation_models=robust_evaluation_models,
            overwrite_run=overwrite_existing_run,
            metric_groups=metric_groups,
            keep_existing_metrics=keep_existing_metrics,
            cpu_optimize=cpu_optimize,
        )

def ifgmi_evaluate_from_wandb(entity, source_project, dest_project, run_id,
                              new_run_id=None, new_run_name=None,
                              overwrite_existing_run=False, metric_groups=None,
                              robust_evaluation_models=None,
                              keep_existing_metrics=False, cpu_optimize=False):

    # Set devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    attack_config, run_name = get_config(entity, source_project, run_id) #, exist_ok=True)

    # set random seed to match attack_config
    torch.manual_seed(attack_config.training.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    target_config, target_weights_path = get_target_config_and_weights(attack_config)
    evaluation_config, evaluation_weights_path = get_evaluation_config_and_weights(attack_config)

    # Load target model
    target_model = get_model(target_config)
    target_model = get_defense(config=target_config, model=target_model)
    target_model.load_model(target_weights_path)
    target_model = ppa_modify.model_compatibility_wrapper(model=target_model, target_config=target_config)

    # Load evaluation model
    evaluation_model = get_model(evaluation_config)
    evaluation_model.load_model(evaluation_weights_path)
    evaluation_model = ppa_modify.model_compatibility_wrapper(model=evaluation_model, target_config=evaluation_config)

    api = wandb.Api()

    # Load real dataset for evaluation comparison
    transform = get_transforms(target_config, train=False)
    target_dataset, _, _ = get_datasets(config=target_config,
                                        train_transform=transform,
                                        test_transform=None)

    # Load idx to class mappings
    dataset_name = target_config.dataset.dataset.lower()
    if dataset_name == 'facescrub':
        idx_to_class = get_facescrub_idx_to_class()
    elif dataset_name == 'stanford_dogs':
        idx_to_class = get_stanford_dogs_idx_to_class()
    else:
        class KeyDict(dict):
            def __missing__(self, key):
                return key
        idx_to_class = KeyDict()

    # Download best_layer ImageFolder from wandb artifact
    artifact = api.artifact(f'{entity}/{source_project}/reconstructions_{run_id}:latest')
    artifact_dir = artifact.download()
    attack_img_dataset_path = os.path.join(artifact_dir, 'best_layer')

    # initialize wandb for new evaluation run (assumes WANDB_API_KEY is set as an environment variable)
    if overwrite_existing_run:
        new_run = wandb.init(project=source_project,
            entity=entity,
            id=run_id,
            resume="must",
        )
    else:
        new_run = wandb.init(project=dest_project,
            entity=entity,
            id=new_run_id,
            name=new_run_name,
            resume="never",
        )
    print(f"wandb initiated with run id: {new_run.id} and run name: {new_run.name}")
    log_target_metadata(new_run, attack_config, target_config)

    eval_transform = get_transforms(evaluation_config, train=False)
    attack_img_dataset = datasets.ImageFolder(attack_img_dataset_path, transform=eval_transform)

    # folder names are zero-padded class indices, so class_to_idx maps them back to original IDs
    folder_to_class = {v: int(k) for k, v in attack_img_dataset.class_to_idx.items()}
    attack_img_dataset.target_transform = folder_to_class.__getitem__
    final_targets = torch.tensor([folder_to_class[t] for t in attack_img_dataset.targets])

    batch_size_single = OmegaConf.select(attack_config, 'attack.eval_batch_size', default=64)
    batch_size = batch_size_single * max(torch.cuda.device_count(), 1)

    result_path = os.path.join("model_inversion", "results", f"{new_run.name}_{new_run.id}")
    os.makedirs(result_path, exist_ok=True)

    evaluate_attack(
        evaluation_model=evaluation_model,
        target_config=target_config,
        attack_img_dataset=attack_img_dataset,
        targets=final_targets,
        device=device,
        gpu_devices=gpu_devices,
        idx_to_class=idx_to_class,
        batch_size=batch_size,
        batch_size_single=batch_size_single,
        run_id=new_run.id,
        target_dataset=target_dataset,
        result_path=result_path,
        wandb_run=new_run,
        metric_groups=metric_groups,
        target_model=target_model,
        robust_evaluation_models=robust_evaluation_models,
        overwrite_run=overwrite_existing_run,
        keep_existing_metrics=keep_existing_metrics,
        cpu_optimize=cpu_optimize,
    )



if __name__ == "__main__":

    entity = "BreuerLab"
    new_run_id = None

    # --- Gather every interactive choice up front, before any run is processed, so the
    # --- batch loop below never stalls on user input partway through. ---

    attack_type, _ = pick(["PPA", "IF-GMI"], "Choose attack type to evaluate:")

    run_mode, _ = pick(
        ["Create a new run for the evaluation results",
         "Update the existing run with evaluation results (overwrite metrics)",
         "Update the existing run, but keep metrics that already have a value (only fill in missing)"],
        "Choose run mode:")
    overwrite_existing_run = run_mode.startswith("Update")
    keep_existing_metrics = "keep metrics" in run_mode

    choice, _ = pick(
        ["No", "Yes"],
        "Enable CPU-side optimizations for the feature-distance metrics "
        "(shared training-dataset build, dataloader workers -- same results, faster)?")
    cpu_optimize = choice == "Yes"

    source_project = "PPA-main" if attack_type == "PPA" else "IF_GMI-main"
    dest_project = "PPA-evaluation" if attack_type == "PPA" else "IF_GMI-evaluation"

    runs_to_evaluate = pick_runs_to_evaluate(entity, source_project)
    if not runs_to_evaluate:
        print("No runs selected, exiting.")
        sys.exit(0)

    common_name = None
    if not overwrite_existing_run:
        common_name = input(
            "Common name for new runs (each will be named '{common_name}_{old_run_name}'): ").strip()


    if attack_type == "PPA":
        option, _ = pick(
            ["Dataset generation", "MIA evaluation (with dataset generation if necessary)"],
            "Choose evaluation method:")
        metric_groups = None if option == "Dataset generation" else pick_metric_groups()

        # Robust evaluation models don't depend on the run being evaluated, so load them
        # once here (only if needed) rather than once per run in the batch below.
        robust_evaluation_models = None
        if metric_groups and (ROBUST_METRIC_GROUPS & metric_groups):
            robust_evaluation_models = pick_robust_evaluation_models(entity)

        for run in runs_to_evaluate:
            new_run_name = None if overwrite_existing_run else f"{common_name}_{run.name}"
            ppa_evaluate_from_wandb(entity=entity, source_project=source_project, dest_project=dest_project,
                                    evaluation_model="stylegan2_ffhq", run_id=run.id, option=option,
                                    new_run_id=new_run_id, new_run_name=new_run_name,
                                    overwrite_existing_run=overwrite_existing_run,
                                    metric_groups=metric_groups,
                                    robust_evaluation_models=robust_evaluation_models,
                                    keep_existing_metrics=keep_existing_metrics,
                                    cpu_optimize=cpu_optimize)
    else:
        metric_groups = pick_metric_groups()

        robust_evaluation_models = None
        if ROBUST_METRIC_GROUPS & metric_groups:
            robust_evaluation_models = pick_robust_evaluation_models(entity)

        for run in runs_to_evaluate:
            new_run_name = None if overwrite_existing_run else f"{common_name}_{run.name}"
            ifgmi_evaluate_from_wandb(entity=entity, source_project=source_project, dest_project=dest_project,
                                      run_id=run.id, new_run_id=new_run_id, new_run_name=new_run_name,
                                      overwrite_existing_run=overwrite_existing_run,
                                      metric_groups=metric_groups,
                                      robust_evaluation_models=robust_evaluation_models,
                                      keep_existing_metrics=keep_existing_metrics,
                                      cpu_optimize=cpu_optimize)