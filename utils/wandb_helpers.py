
import os
from omegaconf import OmegaConf
import wandb
import yaml
import time

def wandb_init_no_config(entity, project, run_id=None, run_name=None):
    # Initiate wandb logger (assumes WANDB_API_KEY is set as an environment variable)
    try:
        run = wandb.init(project=project, 
            entity=entity,
            id=run_id,
            name=run_name,
            resume="allow"
        )

        print(f"wandb initiated with run id: {run.id} and run name: {run.name}")
        return run
    except Exception as e:
        raise FileNotFoundError(f"\nCould not initiate wandb logger\nError: {e}")
    

def wandb_init(config):

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    # Initiate wandb logger (assumes WANDB_API_KEY is set as an environment variable)
    try:
        # project is the name of the project in wandb, entity is the username
        # You can also add tags, group etc.

        run_id = config.training.wandb.run_id if config.training.wandb.run_id else None
        run_name = config.training.wandb.run_name if config.training.wandb.run_name else None

        run = wandb.init(project=config.training.wandb.project, 
            config=OmegaConf.to_container(config), 
            entity=config.training.wandb.entity,
            id=run_id,
            name=run_name,
            resume="allow"
        )
        
        print(f"wandb initiated with run id: {run.id} and run name: {run.name}")
    except Exception as e:
        raise FileNotFoundError(f"\nCould not initiate wandb logger\nError: {e}")
    # log run_name and run_id to wandb_runs.txt
    try:
        with open(os.path.join("utils", "wandb_runs.txt"), "a") as f:
            f.write(f"{run.entity},{run.project},{run.name},{run.id}\n")
    except Exception as e: # try again in 10 seconds
        print(f"Could not write to wandb_runs.txt, trying again in 10 seconds...")
        time.sleep(10)
        with open("wandb_runs.txt", "a") as f:
            f.write(f"{run.name} {run.id}\n")
    return run
        
        
def get_config(entity, project, run_id, print_config=True, use_run_config=True):

    def remove_value_keys(config):
        """Removes the 'value' key from each top-level key in the dictionary if it exists."""
        cleaned_config = {}
        for key, value in config.items():
            # If the current section has a 'value' key, use it; otherwise, keep it as is
            if isinstance(value, dict) and "value" in value:
                cleaned_config[key] = value["value"]

            else:
                cleaned_config[key] = value

        return cleaned_config

    # api = wandb.Api()
    # run = api.run(f"{entity}/{project}/{run_id}")
    run = get_wandb_run(entity, project, run_id)

    if use_run_config:
        # Read the config tracked directly on the run (run.config), bypassing the
        # uploaded 'config.yaml' file, which can be overwritten by manual uploads.
        target_config_data = run.config
        print("Wandb config read from run.config (not downloaded as a file):")
    else:
        # Download the YAML config file
        target_config = run.file("config.yaml").download(replace=True)
        print("Wandb config downloaded as 'config.yaml':")

        with open(target_config.name, 'r') as f:
            target_config_data = yaml.safe_load(f)

    if print_config:
        print("Config contents:", target_config_data)

    target_config = remove_value_keys(target_config_data)
    target_config = OmegaConf.create(target_config)

    return target_config, run.name
    
def get_weights(entity, project, run_id, save_as = None, load_best_model=True):

    # api = wandb.Api()
    # run = api.run(f"{entity}/{project}/{run_id}")
    run = get_wandb_run(entity, project, run_id)

    # filename = save_as if save_as else run.name
    filename = save_as if save_as else run.id
    file_path = f"classifiers/saved_models/{filename}.pth"

    if not load_best_model: # change path to load model from last epoch
        file_path = file_path.replace(".pth", "_last_epoch.pth")

    try:
        target_weights = run.file(file_path).download(replace=True)
    except Exception as e:
        # try with run.name as the filename (for older models)
        print(f"Could not find file {file_path}, trying with run name: {run.name}")
        file_path = file_path.replace(run.id, run.name)
        target_weights = run.file(file_path).download(replace=True)

    del target_weights

    return file_path

def get_wandb_run(entity, project, run_id):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return run

def delete_files_from_project(entity, project, file_to_delete_folder, custom_suffix="run name"):
    api = wandb.Api()

    # iterate through runs in the project
    runs = api.runs(path=f"{entity}/{project}")

    num_files_deleted = 0

    for run in runs:
        if custom_suffix == "run name":
            custom_suffix = run.name
        else:
            raise ValueError(f"Unsupported custom_suffix: {custom_suffix}. Currently supported: 'run name'.")

        file_to_delete = os.path.join(file_to_delete_folder, f"{custom_suffix}.pt")

        try:
            run.file(file_to_delete).delete()
            print(f"Deleted file: {file_to_delete}")
            num_files_deleted += 1
        except Exception as e:
            print(f"Failed to delete file {file_to_delete}: {e}")

    print(f"Total files deleted: {num_files_deleted}")
    return num_files_deleted
