import argparse
import csv
import json
import os
import time
from pick import pick

import wandb

from utils.wandb_helpers import get_config
from experiments.run_commands import run_commands #! relative import
from experiments.all_run_ids import ALL_RUN_IDS, LOAD_BEST_MODEL_FALSE_LIST

# must match attack.optimizer.Adam.lr in configuration/model_inversion/attack/plug_and_play.yaml
DEFAULT_PPA_LR = 0.005

# preset files for saved PPA/IF-GMI settings (targets, attack batch size, attack loss, LR + schedule)
PRESET_FILES = {
    "ppa": os.path.join("experiments", "ppa_presets.csv"),
    "if_gmi": os.path.join("experiments", "if_gmi_presets.csv"),
}
PRESET_FIELDS = {
    "ppa": ["name", "targets", "attack_batch_size", "attack_loss_function",
            "label_smoothing_factor", "lr", "lr_scheduler_milestones", "lr_scheduler_gamma", "align_method", "stdev_spread"],
    # IF-GMI's schedule is per-layer milestones only (null = no scheduler); gamma comes from the attack config
    "if_gmi": ["name", "targets", "attack_batch_size", "attack_loss_function",
               "label_smoothing_factor", "lr", "lr_scheduler_milestones"],
}


def load_presets(experiment_type):
    """Load saved presets for experiment_type ('ppa' or 'if_gmi'), if the preset file exists. Returns {name: {field: value}}."""
    presets_file = PRESET_FILES[experiment_type]
    if not os.path.exists(presets_file):
        return {}
    fields = PRESET_FIELDS[experiment_type]
    presets = {}
    with open(presets_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # .get so presets saved before a field existed load as None
            presets[row["name"]] = {field: json.loads(row[field]) if row.get(field) else None
                                    for field in fields if field != "name"}
    return presets


def save_preset(experiment_type, name, values):
    """Append a new preset (name + field values) to experiment_type's preset file, creating it with a header if needed."""
    presets_file = PRESET_FILES[experiment_type]
    fields = PRESET_FIELDS[experiment_type]
    file_exists = os.path.exists(presets_file)
    os.makedirs(os.path.dirname(presets_file), exist_ok=True)
    if file_exists:
        # migrate a preset file written with an older column set to the current header
        with open(presets_file, newline="") as f:
            reader = csv.DictReader(f)
            old_fields, old_rows = reader.fieldnames, list(reader)
        if old_fields != fields:
            with open(presets_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows({field: row.get(field, "") for field in fields} for row in old_rows)
    with open(presets_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        row = {"name": name}
        row.update({field: json.dumps(values.get(field)) for field in fields if field != "name"})
        writer.writerow(row)


def main(print_only=False):

    title = "Select the experiment type"
    options = ["auto_attack", "ppa", "if_gmi", "hsic", "umap"] # removed "cluster" as an option since we haven't been using it
    experiment_type, _ = pick(options, title)

    # change kwargs as needed based on the experiment type
    commands_kwargs = {}
    if experiment_type == "auto_attack":
        # pick batches
        print("Number of batches to use for AutoAttack (default is 8): ")
        num_batches = input().strip()
        if num_batches != "all":
            try:
                num_batches = int(num_batches)
            except ValueError:
                print("Invalid input, using 8 batches.")
                time.sleep(1)
                num_batches = 8

        # pick split
        title = "Set split on which to run AutoAttack"
        options = ["train", "val", "test"]
        split, _ = pick(options, title)
        print(f"Selected split: {split}")

        # pick adversarial attacks
        title = "Set which attacks to run"
        options = ["apgd_ce", "apgd_t", "fab_t", "square"]
        selected_attacks = pick(options, title, multiselect=True, min_selection_count=1)
        selected_attacks = [option[0] for option in selected_attacks] # get the selected options as a list of strings
        print(f"Selected attacks: {selected_attacks}")

        # pick AutoAttack epsilons
        title = "Set which epsilons to run AutoAttack on"
        options = ["0.00001", "0.0005", "0.0025", "0.031"]
        selected_epsilons = pick(options, title, multiselect=True, min_selection_count=1)
        selected_epsilons = [float(option[0]) for option in selected_epsilons] # get the selected options as a list of floats
        print(f"Selected epsilons: {selected_epsilons}")

        # set commands_kwargs
        commands_kwargs = {"single_attack": False, # use all 3 (4?) attacks by default, can change to single_attack=True to run only one attack at a time
                            "num_batches": num_batches,
                            "split": split,
                            "attacks_to_run": selected_attacks,
                            "epsilons_to_run": selected_epsilons,
                            }
        
    elif experiment_type == "ppa" or experiment_type == "if_gmi":
        dataset_to_10classes = {
            "CelebA": "\"[517, 6, 978, 990, 470, 983, 25, 986, 27, 30]\"",
            "CelebACrop": "\"[517, 6, 978, 990, 470, 983, 25, 986, 27, 30]\"",
            "FaceScrub": "\"[1, 6, 7, 486, 12, 16, 500, 503, 506, 509]\"",
        }
        dataset_to_5classes = {
            "CelebA": "\"[574, 459, 854, 867, 849]\"",
            "CelebACrop": "\"[574, 459, 854, 867, 849]\"",
            "FaceScrub": "\"[306, 107, 56, 327, 57]\"",
        }

        # offer to reuse a saved preset (targets, attack batch size, attack loss, LR [+ schedule for PPA])
        # before asking any of the individual questions below
        preset_label = "PPA" if experiment_type == "ppa" else "IF-GMI"
        used_preset = False
        saved_presets = load_presets(experiment_type)
        if saved_presets:
            title = f"Use a saved {preset_label} preset?"
            options = list(saved_presets.keys()) + ["None (customize manually)"]
            preset_choice, _ = pick(options, title)
            if preset_choice != "None (customize manually)":
                preset = saved_presets[preset_choice]
                mode = "normal"
                targets = preset["targets"]
                commands_kwargs["attack_batch_size"] = preset["attack_batch_size"]
                if preset["attack_loss_function"] is not None:
                    commands_kwargs["attack_loss_function"] = preset["attack_loss_function"]
                if preset["label_smoothing_factor"] is not None:
                    commands_kwargs["label_smoothing_factor"] = preset["label_smoothing_factor"]
                if preset["lr"] is not None:
                    commands_kwargs["lr"] = preset["lr"]
                if experiment_type == "ppa":
                    if preset["lr_scheduler_milestones"] is not None:
                        commands_kwargs["lr_scheduler_milestones"] = preset["lr_scheduler_milestones"]
                    if preset["lr_scheduler_gamma"] is not None:
                        commands_kwargs["lr_scheduler_gamma"] = preset["lr_scheduler_gamma"]
                    if preset["align_method"] is not None:
                        commands_kwargs["align_method"] = preset["align_method"]
                    if preset["stdev_spread"] is not None:
                        commands_kwargs["stdev_spread"] = preset["stdev_spread"]
                else:
                    commands_kwargs["lr_scheduler_milestones"] = preset["lr_scheduler_milestones"]
                    commands_kwargs["use_lr_scheduler"] = preset["lr_scheduler_milestones"] is not None
                print(f"Using {preset_label} preset '{preset_choice}': {preset}")
                used_preset = True

        if not used_preset:
            # get whether to use small/viz/normal attack
            title = "Select attack mode"
            options = ["small", "viz", "viz2", "normal"]
            mode, _ = pick(options, title)

            if mode == "normal":
                # get targets from input
                print(f"Targets for {experiment_type} attack (default is 'all'): ")
                targets = input().strip()
                if targets != "all":
                    try:
                        targets = int(targets)
                    except ValueError:
                        print("WARNING: 'targets' is not int or 'all'.")

                # # get candidate search space size
                # print("Candidate search space size (use 2000 for FaceScrub, 5000 for CelebA):")
                # try:
                #     candidate_search_space_size = input().strip()
                #     candidate_search_space_size = int(candidate_search_space_size)
                # except ValueError:
                #     print("Invalid input, using 2000 candidates.")
                #     time.sleep(1)
                #     candidate_search_space_size = 2000

                # set intermediate steps for if_gmi attack
                if experiment_type == "if_gmi":
                    title = "Set intermediate steps for IF-GMI attack (first option for FaceScrub, second for CelebA, Third for Stanford Dogs)"
                    options = ['[50,10,10,10]', '[70,25,25,25]', '[100,10,10,10,10,10,10]']
                    intermediate_steps, _ = pick(options, title)
                    commands_kwargs["intermediate_steps"] = intermediate_steps
            elif mode == "viz":
                targets = dataset_to_10classes

                if experiment_type == "if_gmi":
                    title = "Set intermediate steps for IF-GMI attack (first option for FaceScrub, second for CelebA, Third for Stanford Dogs)"
                    options = ['[50,10,10,10]', '[70,25,25,25]', '[100,10,10,10,10,10,10]']
                    intermediate_steps, _ = pick(options, title)
                    commands_kwargs["intermediate_steps"] = intermediate_steps

            elif mode == "viz2":
                targets = dataset_to_5classes

                if experiment_type == "if_gmi":
                    title = "Set intermediate steps for IF-GMI attack (first option for FaceScrub, second for CelebA, Third for Stanford Dogs)"
                    options = ['[50,10,10,10]', '[70,25,25,25]', '[100,10,10,10,10,10,10]']
                    intermediate_steps, _ = pick(options, title)
                    commands_kwargs["intermediate_steps"] = intermediate_steps

            else: # small
                targets = None
                candidate_search_space_size = None
                if experiment_type == "if_gmi":
                    commands_kwargs["intermediate_steps"] = None
            # import pdb; pdb.set_trace()
        elif experiment_type == "if_gmi":
            # preset forces mode="normal" but doesn't store intermediate steps (dataset-dependent), so still ask for it
            title = "Set intermediate steps for IF-GMI attack (first option for FaceScrub, second for CelebA, Third for Stanford Dogs)"
            options = ['[50,10,10,10]', '[70,25,25,25]', '[100,10,10,10,10,10,10]']
            intermediate_steps, _ = pick(options, title)
            commands_kwargs["intermediate_steps"] = intermediate_steps

        # set more kwargs
        dataset_to_eval_model = {
            "FaceScrub": "3ivev4xr",
            "CelebACrop": "sufkoopo",
            "CelebA": "l86pwqbg",
            "stanford_dogs": "j2iwjopb",
        }

        dataset_to_candidate_search_space_size = {
            "FaceScrub": 2000,
            "CelebACrop": 5000,
            "CelebA": 5000,
            "stanford_dogs": 2000,
        }

        if experiment_type == "ppa":
            dataset_to_attack_config = {
                "FaceScrub": "plug_and_play",
                "CelebACrop": "plug_and_play",
                "CelebA": "plug_and_play",
                "stanford_dogs": "plug_and_play_dogs",
            }
        elif experiment_type == "if_gmi":
            dataset_to_attack_config = {
                "FaceScrub": "if_gmi",
                "CelebACrop": "if_gmi",
                "CelebA": "if_gmi",
                "stanford_dogs": "if_gmi_dogs",
            }

        if not used_preset:
            # set IF-GMI attack batch size
            if experiment_type == "if_gmi":
                    print("Set attack batch size (default is 16): ")
                    attack_batch_size = input().strip()
                    try:
                        attack_batch_size = int(attack_batch_size)
                    except ValueError:
                        print("Invalid input, using 16 as attack batch size.")
                        time.sleep(1)
                        attack_batch_size = 16
                    commands_kwargs["attack_batch_size"] = attack_batch_size

            # set PPA attack batch size
            if experiment_type == "ppa":
                    print("Set attack batch size (default is 25): ")
                    attack_batch_size = input().strip()
                    try:
                        attack_batch_size = int(attack_batch_size)
                    except ValueError:
                        print("Invalid input, using 25 as attack batch size.")
                        time.sleep(1)
                        attack_batch_size = 25
                    commands_kwargs["attack_batch_size"] = attack_batch_size

            # set attack loss function (option to use cross-entropy or negative label smoothing CrossEntropy loss) -- PPA and IF-GMI
            if experiment_type in ("ppa", "if_gmi"):
                    title = f"Select attack loss function for {preset_label} optimization"
                    options = ["poincare (default)", "cross_entropy", "negative_label_smoothing"]
                    attack_loss_choice, _ = pick(options, title)
                    if attack_loss_choice == "cross_entropy":
                        commands_kwargs["attack_loss_function"] = "cross_entropy"
                    elif attack_loss_choice == "negative_label_smoothing":
                        commands_kwargs["attack_loss_function"] = "negative_label_smoothing"
                        print("Label smoothing factor (default is -0.1): ")
                        label_smoothing_factor = input().strip()
                        try:
                            label_smoothing_factor = float(label_smoothing_factor)
                        except ValueError:
                            print("Invalid input, using -0.1 as label smoothing factor.")
                            time.sleep(1)
                            label_smoothing_factor = -0.1
                        commands_kwargs["label_smoothing_factor"] = label_smoothing_factor

            # set PPA learning rate schedule: initial LR + MultiStepLR milestones/gamma (scheduler always on;
            # defaults are lr=0.005 (0.01 for stanford_dogs), milestones=[30,40], gamma=0.1)
            if experiment_type == "ppa":
                    title = "Customize the learning rate schedule for PPA optimization? (default: lr per attack config, milestones=[30, 40], gamma=0.1)"
                    options = ["Use default", "Customize"]
                    lr_scheduler_choice, _ = pick(options, title)
                    if lr_scheduler_choice == "Customize":
                        print("Initial learning rate (default is attack config's default, e.g. 0.005): ")
                        lr_input = input().strip()
                        if lr_input != "":
                            try:
                                lr = float(lr_input)
                                commands_kwargs["lr"] = lr
                            except ValueError:
                                print("Invalid input, using the attack config's default learning rate.")
                                time.sleep(1)

                        print("MultiStepLR milestones as comma-separated epochs (default 30,40): ")
                        milestones_input = input().strip()
                        try:
                            lr_scheduler_milestones = [int(m.strip()) for m in milestones_input.split(",") if m.strip() != ""]
                            if not lr_scheduler_milestones:
                                raise ValueError
                        except ValueError:
                            print("Invalid input, using default milestones [30, 40].")
                            time.sleep(1)
                            lr_scheduler_milestones = [30, 40]
                        commands_kwargs["lr_scheduler_milestones"] = lr_scheduler_milestones

                        print("MultiStepLR gamma (default 0.1): ")
                        gamma_input = input().strip()
                        try:
                            lr_scheduler_gamma = float(gamma_input)
                        except ValueError:
                            print("Invalid input, using default gamma 0.1.")
                            time.sleep(1)
                            lr_scheduler_gamma = 0.1
                        commands_kwargs["lr_scheduler_gamma"] = lr_scheduler_gamma

            # set IF-GMI initial learning rate (no LR scheduler for IF-GMI, unlike PPA)
            if experiment_type == "if_gmi":
                    title = "Customize the initial learning rate for IF-GMI optimization? (default: attack config's default, e.g. 0.005)"
                    options = ["Use default", "Customize"]
                    lr_choice, _ = pick(options, title)
                    if lr_choice == "Customize":
                        print("Initial learning rate (default is attack config's default, e.g. 0.005): ")
                        lr_input = input().strip()
                        if lr_input != "":
                            try:
                                lr = float(lr_input)
                                commands_kwargs["lr"] = lr
                            except ValueError:
                                print("Invalid input, using the attack config's default learning rate.")
                                time.sleep(1)

                    # optional per-layer MultiStepLR schedule (milestones are iterations within each layer)
                    title = "Use a learning rate schedule for IF-GMI optimization?"
                    options = ["No", "Yes"]
                    use_lr_schedule, _ = pick(options, title)
                    if use_lr_schedule == "Yes":
                        title = "Select per-layer LR schedule milestones (one list per layer; gamma per attack config, default 0.32)"
                        options = ["[[30,40], [5], [5], [5]]",
                                   "[[10,20,30,40], [5], [5], [5]]",
                                   "[[40,55], [15], [15], [15]]",
                                   "[[20,30,40,55], [15,20], [15,20], [15,20]]"]
                        milestones_choice, _ = pick(options, title)
                        commands_kwargs["lr_scheduler_milestones"] = json.loads(milestones_choice)
                    commands_kwargs["use_lr_scheduler"] = use_lr_schedule == "Yes"

            # set PPA align method (version of AlignMI used)
            if experiment_type == "ppa":
                    title = "Select PPA align method"
                    options = ["old", "vanilla", "PAA", "TAA"]
                    align_method, _ = pick(options, title)
                    commands_kwargs["align_method"] = align_method
                    if align_method == "PAA":
                        print("PAA stdev_spread (default is 0.1): ")
                        stdev_spread = input().strip()
                        try:
                            stdev_spread = float(stdev_spread)
                        except ValueError:
                            print("Invalid input, using 0.1 as stdev_spread.")
                            time.sleep(1)
                            stdev_spread = 0.1
                        commands_kwargs["stdev_spread"] = stdev_spread

            # after gathering targets, attack batch size, attack loss, label smoothing factor, and LR (+ milestones/gamma
            # for PPA), offer to save these settings as a reusable preset
            if experiment_type in ("ppa", "if_gmi"):
                save_preset_choice = input(f"Save these {preset_label} settings as a preset? (y/n): ").strip().lower()
                if save_preset_choice.startswith("y"):
                    preset_name = input("Enter a name for this preset: ").strip()
                    preset_values = {
                        "targets": targets,
                        "attack_batch_size": commands_kwargs.get("attack_batch_size"),
                        "attack_loss_function": commands_kwargs.get("attack_loss_function"),
                        "label_smoothing_factor": commands_kwargs.get("label_smoothing_factor"),
                        "lr": commands_kwargs.get("lr"),
                    }
                    if experiment_type == "ppa":
                        preset_values["lr_scheduler_milestones"] = commands_kwargs.get("lr_scheduler_milestones")
                        preset_values["lr_scheduler_gamma"] = commands_kwargs.get("lr_scheduler_gamma")
                        preset_values["align_method"] = commands_kwargs.get("align_method")
                        preset_values["stdev_spread"] = commands_kwargs.get("stdev_spread")
                    else:
                        preset_values["lr_scheduler_milestones"] = (
                            commands_kwargs.get("lr_scheduler_milestones") if commands_kwargs.get("use_lr_scheduler") else None)
                    save_preset(experiment_type, preset_name, preset_values)
                    print(f"Saved preset '{preset_name}'.")


        commands_kwargs.update({
            "targets": targets,
            "dataset_to_eval_model": dataset_to_eval_model, # dictionary mapping dataset names to evaluation model run ids,
            "dataset_to_attack_config": dataset_to_attack_config, # dictionary mapping dataset names to attack config names,
            # "dataset_to_stylegan_path": dataset_to_stylegan_path, # dictionary mapping dataset names to stylegan model paths,
            "dataset_to_candidate_search_space_size": dataset_to_candidate_search_space_size, # dictionary mapping dataset names to candidate search space sizes,
            "mode": mode # one of "small", "viz", "viz2", "normal" -- straight from the picker
        })



    elif experiment_type == "cluster":
        pass
    elif experiment_type == "hsic":
        pass
    elif experiment_type == "umap":
        pass
    else:
        raise ValueError(f"Experiment type {experiment_type} not recognized. Supported: 'auto_attack', 'ppa', 'if_gmi', 'cluster', 'hsic', 'umap'.")

    # set load_best_model_false_list
    commands_kwargs["load_best_model_false_list"] = LOAD_BEST_MODEL_FALSE_LIST

    # Step 1: select which run id groups to draw from
    title = "Select which run ID groups to include"
    selected_dicts = pick(list(ALL_RUN_IDS.keys()), title, multiselect=True, min_selection_count=1)
    selected_dict_names = [option[0] for option in selected_dicts]

    # Build the pool of runs from the selected groups
    pooled_run_ids = {}
    for dict_name in selected_dict_names:
        pooled_run_ids.update(ALL_RUN_IDS[dict_name])

    # Step 2: select specific runs within the pooled options
    title = "Select the runs to include in the experiment"
    run_options = list(pooled_run_ids.keys()) + ["All runs"]
    selected = pick(run_options, title, multiselect=True, min_selection_count=1)
    selected_options = [option[0] for option in selected]
    if "All runs" in selected_options:
        run_ids = pooled_run_ids
    else:
        run_ids = {option: pooled_run_ids[option] for option in selected_options}

    # find which dataset each model corresponds to
    model_name_to_dataset = {}
    for model_name, model_run_id in run_ids.items():
        model_run = wandb.Api().run(f"BreuerLab/ModelInversion/{model_run_id}")
        model_run_config, _ = get_config(entity=model_run.entity,
                                      project=model_run.project,
                                      run_id=model_run.id,
                                      print_config=False)
        dataset = model_run_config.dataset.dataset
        if dataset == "CelebA" and model_run_config.dataset.use_hdcrop:
            dataset = "CelebACrop"
        model_name_to_dataset[model_name] = dataset

    # get commands for the selected runs
    commands = get_commands(run_ids,
                            # experiment_name,
                            experiment_type,
                            model_name_to_dataset,
                            **commands_kwargs
                            )

    # print the commands instead of running them if --print was passed
    if print_only:
        for attack_run_name, cmd in commands.items():
            print(f"# {attack_run_name}\n{cmd}\n")
        return

    # run the commands
    run_commands(commands)


def get_commands(run_ids, experiment_type, model_name_to_dataset=None, **kwargs):
    commands = {}
    load_best_model_false_list = kwargs.get('load_best_model_false_list', [])

    if experiment_type == "auto_attack":
        split = kwargs.get('split', "val")

        use_rand_for_mid = input("If you're using MID models, they might not be deterministic. Use 'rand' for MID AutoAttack? (y/n)")

        for model_name, run_id in run_ids.items():
            load_best_model = not (model_name in load_best_model_false_list)
            eps_list = kwargs.get('epsilons_to_run')
            num_batches = kwargs.get('num_batches', 8)

            if use_rand_for_mid.lower().startswith('y') and "MID" in model_name:
                version = "rand"
            else:
                version = "standard"

            attacks_to_run = kwargs.get('attacks_to_run')

            for eps in eps_list:
                custom_name = f"{split}_eps{eps}"
                attack_run_name = f"{model_name}__{custom_name}"
                command = get_auto_attack_command(run_id, attack_run_name, eps, version,
                                                run_apgd_ce='apgd_ce' in attacks_to_run,
                                                run_apgd_t='apgd_t' in attacks_to_run,
                                                run_fab_t='fab_t' in attacks_to_run,
                                                run_square='square' in attacks_to_run,
                                                split=split,
                                                load_best_model=load_best_model,
                                                num_batches=num_batches,
                                                custom_name=custom_name,
                                                )
                commands[attack_run_name] = command

    elif experiment_type == "ppa" or experiment_type == "if_gmi":
        targets = kwargs.get('targets', "all") # targets for inversion attack, default is "all"
        dataset_to_eval_model = kwargs.get('dataset_to_eval_model', None)
        # dataset_to_stylegan_path = kwargs.get('dataset_to_stylegan_path', None)
        dataset_to_attack_config = kwargs.get('dataset_to_attack_config', None)
        dataset_to_candidate_search_space_size = kwargs.get('dataset_to_candidate_search_space_size', None)
        
        
        assert dataset_to_eval_model is not None, f"dataset_to_eval_model must be provided for {experiment_type} attack commands"
        assert model_name_to_dataset is not None, f"model_name_to_dataset must be provided for {experiment_type} attack commands"
        # assert dataset_to_stylegan_path is not None, f"dataset_to_stylegan_path must be provided for {experiment_type} attack commands"
        assert dataset_to_attack_config is not None, f"dataset_to_attack_config must be provided for {experiment_type} attack commands"
        assert dataset_to_candidate_search_space_size is not None, f"dataset_to_candidate_search_space_size must be provided for {experiment_type} attack commands"
        
        mode = kwargs.get('mode', "normal") # attack mode straight from the picker: "small", "viz", or "normal"
        small = (mode == "small") # whether to use small attack for testing purposes
        # candidate_search_space_size = kwargs.get('candidate_search_space_size', 2000) # default is 2000 candidates

        # loss suffix for custom names: 'CE' for cross entropy, 'NLS-CE' for negative label smoothing, nothing for Poincare (default)
        attack_loss_function = kwargs.get('attack_loss_function', None)
        if attack_loss_function == "cross_entropy":
            loss_suffix = "_CE"
        elif attack_loss_function == "negative_label_smoothing":
            loss_suffix = "_NLS-CE"
        else:
            loss_suffix = ""

        for model_name, run_id in run_ids.items():
            load_best_model = not (model_name in load_best_model_false_list)
            current_dataset = model_name_to_dataset[model_name]

            # for viz mode, targets is a dataset_to_10classes dict -- resolve it per-model's dataset
            if isinstance(targets, dict):
                model_targets = targets[current_dataset]
            else:
                model_targets = targets

            if mode == "viz":
                custom_name = f"{experiment_type}_viz"
            elif mode == "viz2":
                custom_name = f"{experiment_type}_viz2"
            elif isinstance(model_targets, str) and model_targets != "all":
                custom_name = f"{experiment_type}_manyT"
            elif model_targets == None and small:
                custom_name = f"{experiment_type}_smallT"
            else:
                custom_name = f"{experiment_type}_{model_targets}T"
            custom_name = custom_name.replace("ppa_", "") + loss_suffix
            attack_run_name = f"{model_name}__{custom_name}"

            evaluation_model_id = dataset_to_eval_model[current_dataset] # get the evaluation model id for the current dataset
            # stylegan_path = dataset_to_stylegan_path[current_dataset] # get the stylegan model path for the current dataset
            attack_config = dataset_to_attack_config[current_dataset]
            candidate_search_space_size = dataset_to_candidate_search_space_size[current_dataset]

            # get command based on which inversion attack is used
            if experiment_type == "ppa":
                attack_batch_size = kwargs.get('attack_batch_size', 25)
                attack_loss_function = kwargs.get('attack_loss_function', None)
                label_smoothing_factor = kwargs.get('label_smoothing_factor', None)
                lr = kwargs.get('lr', None)
                lr_scheduler_milestones = kwargs.get('lr_scheduler_milestones', None)
                lr_scheduler_gamma = kwargs.get('lr_scheduler_gamma', None)
                align_method = kwargs.get('align_method', None)
                stdev_spread = kwargs.get('stdev_spread', None) if align_method == "PAA" else None

                # flag PAA/TAA in the custom/run name (PAA also gets its stdev_spread, e.g. 0.001 -> "_PAA.001")
                if align_method in ("PAA", "TAA"):
                    custom_name = f"{custom_name}_{align_method}"
                    if align_method == "PAA" and stdev_spread is not None:
                        custom_name += f"{stdev_spread:g}".removeprefix("0")
                    attack_run_name = f"{model_name}__{custom_name}"

                # flag non-default LR in the custom/run name (mirrors scripts/append_lr_to_custom_name.py,
                # which applies the same suffix retroactively to existing PPA-main wandb runs)
                if lr is not None and lr != DEFAULT_PPA_LR:
                    custom_name = f"{custom_name}_LR{lr}"
                    attack_run_name = f"{model_name}__{custom_name}"

                command = get_ppa_command(run_id, evaluation_model_id, attack_run_name, model_targets,
                                        candidate_search_space_size=candidate_search_space_size,
                                        load_best_model=load_best_model,
                                        small=small,
                                        attack_config=attack_config,
                                        attack_batch_size=attack_batch_size,
                                        attack_loss_function=attack_loss_function,
                                        label_smoothing_factor=label_smoothing_factor,
                                        lr=lr,
                                        lr_scheduler_milestones=lr_scheduler_milestones,
                                        lr_scheduler_gamma=lr_scheduler_gamma,
                                        align_method=align_method,
                                        stdev_spread=stdev_spread,
                                        custom_name=custom_name,)
            elif experiment_type == "if_gmi":
                intermediate_steps = kwargs.get('intermediate_steps', None)

                attack_batch_size = kwargs.get('attack_batch_size', 16)
                label_smoothing_factor = kwargs.get('label_smoothing_factor', None)
                lr = kwargs.get('lr', None)
                use_lr_scheduler = kwargs.get('use_lr_scheduler', None) # None = leave attack config's default
                lr_scheduler_milestones = kwargs.get('lr_scheduler_milestones', None)
                command = get_ifgmi_command(run_id, evaluation_model_id, attack_run_name, model_targets,
                                        candidate_search_space_size=candidate_search_space_size,
                                        intermediate_steps=intermediate_steps,
                                        attack_batch_size=attack_batch_size,
                                        load_best_model=load_best_model,
                                        small=small,
                                        attack_config=attack_config,
                                        attack_loss_function=attack_loss_function,
                                        label_smoothing_factor=label_smoothing_factor,
                                        lr=lr,
                                        use_lr_scheduler=use_lr_scheduler,
                                        lr_scheduler_milestones=lr_scheduler_milestones,
                                        custom_name=custom_name,)
            commands[attack_run_name] = command

    elif experiment_type == "cluster":
        split = kwargs.get('split', "val") # default split is "val"
        num_batches = kwargs['num_batches']

        for model_name, run_id in run_ids.items():
            load_best_model = not (model_name in load_best_model_false_list)
            custom_name = f"{split}_{num_batches}B"
            cluster_run_name = f"{model_name}__{custom_name}"
            command = get_cluster_command(run_id, cluster_run_name,
                            load_best_model, split=split, num_batches=num_batches, custom_name=custom_name)
            commands[cluster_run_name] = command
    
    elif experiment_type == "hsic":
        if "split" in kwargs:
            split = kwargs['split']
        else:
            split = "train" # if "test" not in experiment_name.lower() else "val" # default split is "train", but use "val" for testing purposes
        for model_name, run_id in run_ids.items():
            load_best_model = not (model_name in load_best_model_false_list)
            custom_name = split
            hsic_run_name = f"{model_name}__{custom_name}"
            command = get_hsic_command(run_id, hsic_run_name,
                            load_best_model, split=split, custom_name=custom_name)
            commands[hsic_run_name] = command

    elif experiment_type == "umap":
        if "split" in kwargs:
            split = kwargs['split']
        else:
            split = "train"

        for model_name, run_id in run_ids.items():
            load_best_model = not (model_name in load_best_model_false_list)
            custom_name = split
            umap_run_name = f"{model_name}__{custom_name}"
            command = get_umap_command(run_id, umap_run_name,
                            load_best_model, split=split, custom_name=custom_name)
            commands[umap_run_name] = command

    else:
        raise ValueError(f"Experiment type {experiment_type} not recognized. Supported: auto_attack, ppa, cluster, hsic")
    
    return commands

def get_auto_attack_command(target_wandb_id, run_name, eps, version, run_apgd_ce, run_apgd_t, run_fab_t, run_square, split, load_best_model,
            num_batches="all", run_individual_attacks=True, custom_name=None):
    command = f"python evaluate_classifier.py training.wandb.project_to_load_from=ModelInversion target_wandb_id={target_wandb_id} \
training.wandb.track=True training.wandb.entity=BreuerLab training.wandb.run_name={run_name} auto_attack.calculate=True dataset.normalize=False \
training.wandb.project=auto_attack auto_attack.eps={eps} auto_attack.num_batches={num_batches} auto_attack.run_apgd_ce={run_apgd_ce} \
auto_attack.run_apgd_t={run_apgd_t} auto_attack.run_fab_t={run_fab_t} auto_attack.run_square={run_square} auto_attack.split={split} \
training.load_best_model={load_best_model} auto_attack.run_individual_attacks={run_individual_attacks} auto_attack.version={version} custom_name={custom_name}"
    return command

def get_ppa_command(target_wandb_id, evaluation_model_id, attack_run_name, targets,
                    candidate_search_space_size=2000, attack_batch_size=25,
                    load_best_model=True, small=False, attack_config=None,
                    attack_loss_function=None, label_smoothing_factor=None,
                    lr=None, lr_scheduler_milestones=None, lr_scheduler_gamma=None, align_method=None, stdev_spread=None, custom_name=None):
    # optional override for the PPA optimization attack loss (e.g. negative label smoothing CrossEntropy)
    extra_overrides = ""
    if attack_loss_function is not None:
        extra_overrides += f" attack.attack.attack_loss_function={attack_loss_function}"
        if label_smoothing_factor is not None:
            extra_overrides += f" attack.attack.label_smoothing_factor={label_smoothing_factor}"

    if align_method is not None:
        extra_overrides += f" attack.attack.align_method={align_method}"
    if stdev_spread is not None:
        extra_overrides += f" attack.attack.PAA.stdev_spread={stdev_spread}"

    # optional override for the PPA learning rate schedule: initial LR (Adam) + MultiStepLR milestones/gamma; scheduler itself is always on
    if lr is not None:
        extra_overrides += f" attack.attack.optimizer.Adam.lr={lr}"
    if lr_scheduler_milestones is not None:
        milestones_str = "[" + ",".join(str(m) for m in lr_scheduler_milestones) + "]"
        extra_overrides += f" attack.attack.lr_scheduler.MultiStepLR.milestones={milestones_str}"
    if lr_scheduler_gamma is not None:
        extra_overrides += f" attack.attack.lr_scheduler.MultiStepLR.gamma={lr_scheduler_gamma}"

    if small: # for quick testing purposes
        command = f"python run_model_inversion.py target_wandb_id={target_wandb_id} attack={attack_config} training.wandb.run_name={attack_run_name} \
attack.evaluation_model.wandb_id={evaluation_model_id} training.wandb.track=True training.wandb.project=PPA-testing \
training.wandb.target_project=ModelInversion training.wandb.evaluation_project=ModelInversion training.wandb.entity=BreuerLab \
training.load_best_model={load_best_model} attack.attack.targets=2 attack.attack.num_epochs=2 \
attack.candidates.num_candidates=8 attack.candidates.candidate_search.search_space_size=50 attack.final_selection.samples_per_target=2 \
custom_name={custom_name}{extra_overrides}"
    else:
        command =  f"python run_model_inversion.py target_wandb_id={target_wandb_id} attack={attack_config} training.wandb.run_name={attack_run_name} \
attack.evaluation_model.wandb_id={evaluation_model_id} attack.attack.targets={targets} training.wandb.track=True training.wandb.project=PPA-main \
training.wandb.target_project=ModelInversion training.wandb.evaluation_project=ModelInversion training.wandb.entity=BreuerLab \
training.load_best_model={load_best_model} attack.candidates.candidate_search.search_space_size={candidate_search_space_size} \
attack.candidates.candidate_search.batch_size={attack_batch_size} attack.attack.batch_size={attack_batch_size} custom_name={custom_name}{extra_overrides}"
    return command

def get_ifgmi_command(target_wandb_id, evaluation_model_id, attack_run_name, targets,
                    intermediate_steps=None,
                    attack_batch_size=16,
                    attack_eval_batch_size=128,
                    candidate_search_space_size=None,
                    load_best_model=True, small=False,
                    attack_config=None,
                    attack_loss_function=None, label_smoothing_factor=None, lr=None,
                    use_lr_scheduler=None, lr_scheduler_milestones=None,
                    custom_name=None):
    # optional override for the IF-GMI optimization attack loss (e.g. negative label smoothing CrossEntropy)
    extra_overrides = ""
    if attack_loss_function is not None:
        extra_overrides += f" attack.attack.attack_loss_function={attack_loss_function}"
        if label_smoothing_factor is not None:
            extra_overrides += f" attack.attack.label_smoothing_factor={label_smoothing_factor}"

    # optional override for the IF-GMI initial learning rate (Adam)
    if lr is not None:
        extra_overrides += f" attack.attack.optimizer.Adam.lr={lr}"

    # optional per-layer LR schedule: use flag + milestones (list of lists, one per layer). The picker options have
    # 4 entries (FaceScrub/CelebA layer count), so pad with the last entry if intermediate_steps has more layers.
    if use_lr_scheduler is not None:
        extra_overrides += f" attack.attack.lr_scheduler.use={str(use_lr_scheduler).lower()}"
    if use_lr_scheduler and lr_scheduler_milestones is not None:
        num_layers = len(json.loads(intermediate_steps)) if intermediate_steps else len(lr_scheduler_milestones)
        milestones = lr_scheduler_milestones + [lr_scheduler_milestones[-1]] * (num_layers - len(lr_scheduler_milestones))
        milestones_str = json.dumps(milestones, separators=(",", ":"))
        extra_overrides += f" attack.attack.lr_scheduler.MultiStepLR.milestones='{milestones_str}'"

    if small: # for quick testing purposes
        command = f"python run_model_inversion.py target_wandb_id={target_wandb_id} attack={attack_config} training.wandb.run_name={attack_run_name} \
attack.evaluation_model.wandb_id={evaluation_model_id} training.wandb.track=True training.wandb.project=IF_GMI-main \
training.wandb.target_project=ModelInversion training.wandb.evaluation_project=ModelInversion training.wandb.entity=BreuerLab \
training.load_best_model={load_best_model} attack.attack.targets='[2,3]' attack.intermediate.steps='[2,1,1,1]' \
attack.candidates.num_candidates=8 attack.candidates.candidate_search.search_space_size=50 custom_name={custom_name}{extra_overrides}"
    else:
        command = f"python run_model_inversion.py target_wandb_id={target_wandb_id} attack={attack_config} training.wandb.run_name={attack_run_name} \
attack.evaluation_model.wandb_id={evaluation_model_id} training.wandb.track=True training.wandb.project=IF_GMI-main \
training.wandb.target_project=ModelInversion training.wandb.evaluation_project=ModelInversion training.wandb.entity=BreuerLab \
training.load_best_model={load_best_model} attack.attack.targets={targets} attack.intermediate.steps={intermediate_steps} \
attack.candidates.candidate_search.search_space_size={candidate_search_space_size} attack.attack.batch_size={attack_batch_size} \
attack.attack.eval_batch_size={attack_eval_batch_size} custom_name={custom_name}{extra_overrides}"
    return command

def get_cluster_command(target_wandb_id, cluster_run_name, load_best_model=True, split="train", num_batches="all", custom_name=None):
    command = f"python evaluate_classifier.py training.wandb.track=True training.wandb.entity=BreuerLab training.wandb.project_to_load_from=ModelInversion \
training.wandb.project=cluster training.wandb.run_name={cluster_run_name} cluster_metrics.calculate=True target_wandb_id={target_wandb_id} \
training.load_best_model={load_best_model} cluster_metrics.split={split} cluster_metrics.num_batches={num_batches} custom_name={custom_name}"
    return command

def get_hsic_command(target_wandb_id, run_name, load_best_model=True, split="train", custom_name=None):
    command = f"python evaluate_classifier.py training.wandb.track=True training.wandb.entity=BreuerLab training.wandb.project_to_load_from=ModelInversion \
training.wandb.project=hsic hsic.calculate=True target_wandb_id={target_wandb_id} training.wandb.run_name={run_name} hsic.split={split} \
training.load_best_model={load_best_model} custom_name={custom_name}"
    return command

def get_umap_command(target_wandb_id, run_name, load_best_model=True, split="train", custom_name=None):
    command = f"python evaluate_classifier.py training.wandb.track=True training.wandb.entity=BreuerLab training.wandb.project_to_load_from=ModelInversion \
training.wandb.project=umap umap.calculate=True target_wandb_id={target_wandb_id} training.wandb.run_name={run_name} \
umap.split={split} training.load_best_model={load_best_model} custom_name={custom_name}"
    return command

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", dest="print_only", action="store_true", default=False,
                        help="print the generated commands instead of running them")
    args = parser.parse_args()
    main(print_only=args.print_only)