# evaluate classifiers loaded from wandb

import hydra
import wandb
import os
import torch
import time
import pickle
import pandas as pd
import omegaconf
from omegaconf import OmegaConf, open_dict
import matplotlib.pyplot as plt
import torch.nn as nn

from data_processing.data_loaders import get_data_loaders
from classifiers.get_model import get_model
from defenses.get_defense import get_defense

from utils import wandb_helpers
from utils.fisher import diag_fisher
from utils.hcr_bounds import calculate_hcr
from utils.bido_metrics import calculate_hsic
from utils.auto_attack import auto_attack, normalize_model
from utils.calculate_umap import calculate_umap
from utils.cluster_metrics import run_cluster_metrics


@hydra.main(config_path="configuration/classifier", config_name="config_evaluate.yaml", version_base="1.3")
def evaluate_classifier(config):
    
    if torch.cuda.is_available() and config.training.device != "cuda":
        question = f"\nCuda is available but not configured from command to be used! Do you wish to use cuda instead of {config.training.device}?\nType y to use cuda, enter if not:"
        
        use_cuda = input(question)
        
        if use_cuda.lower().strip() == "y":
            config.training.device = 'cuda'
            
    if config.training.wandb.track:
        wandb_helpers.wandb_init(config)
        # try:
            # if len(config.training.wandb.run_name.split("__")) == 3: # the wandb run name is in our experiment format: model_name__custom_name__experiment_name
            #     model_name, custom_name, experiment_name = config.training.wandb.run_name.split("__")
            #     wandb.log({
            #         "model_name": model_name,
            #         "custom_name": custom_name,
            #         "experiment_name": experiment_name,
            #     })
        # except AttributeError: #! Remove before daredeval v2.0
        #     print("Warning: wandb run name not in expected format, skipping logging of model_name, custom_name, and experiment_name")


    # get config of model we want to evaluate
    target_config, _ = wandb_helpers.get_config(entity=config.training.wandb.entity,
                                                project=config.training.wandb.project_to_load_from,
                                                run_id=config.target_wandb_id)
    
    # use the target defense and model config
    config.defense = target_config.defense
    config.model = target_config.model

    # use most of the dataset config from the target model, but keep the current value of 'normalize'
    normalize = config.dataset.normalize
    config.dataset = target_config.dataset
    with open_dict(config.dataset):
        config.dataset.normalize = normalize

    if config.training.wandb.track:
        wandb.log({
            "defense": config.defense.name,
            "architecture": config.model.architecture,
            "dataset": config.dataset.dataset,
        })
    
    # Load the data
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # Load the model
    model = get_model(config)
    
    # Load defense
    model = get_defense(config=config, model=model)

    if config.dataset.normalize:
        normalized_model = model # the normalization will happen in the data pre-processing, no need to change the model
    else:
        # Add normalization before rest of forward pass in model
        normalized_model = normalize_model(model, config, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], device=config.training.device)

    # Load trained model weights
    assert config.target_wandb_id, "Please provide a run id to load the model from for evaluation"
    try:
        load_best_model = config.training.load_best_model
    except omegaconf.errors.ConfigAttributeError:
        load_best_model = True
        print("Warning: load_best_model not found in config, defaulting to True")
    
    model_weights_path = wandb_helpers.get_weights(
                                                    entity   = config.training.wandb.entity,
                                                    project  = config.training.wandb.project_to_load_from,
                                                    run_id = config.target_wandb_id,
                                                    load_best_model=load_best_model,
    )
    
    try:
        normalized_model.load_model(model_weights_path)
    except Exception as e:
        print("Error loading model weights. Trying again in 10 seconds...")
        time.sleep(10)
        normalized_model.load_model(model_weights_path)
    print(f"Model weights loaded from {model_weights_path}")

    if config.accuracy.calculate:
        split = config.accuracy.split
        if split == "train":
            print(">> Evaluating on train set...")
            accuracy_loader = train_loader
        elif split == "val":
            print(">> Evaluating on val set...")
            accuracy_loader = val_loader
        elif split == "test":
            print(">> Evaluating on test set...")
            accuracy_loader = test_loader
        else:
            raise ValueError(f"Unknown split: {split}. Please choose from 'train', 'val', or 'test'.")

        loss, accuracy = normalized_model.evaluate(accuracy_loader)
        top5_accuracy = calculate_top5_accuracy(normalized_model, accuracy_loader, device=config.training.device)
    
        print(f"{split} Loss: {loss}, {split} Accuracy: {accuracy}, {split} Top-5 Accuracy: {top5_accuracy}")
    
        if config.training.wandb.track:
            wandb.log({f"{split}_loss": loss})
            wandb.log({f"{split}_accuracy": accuracy})
            wandb.log({f"{split}_top5_accuracy": top5_accuracy})

    if config.hsic.calculate:
        # make sure hsic directory exists
        hsic_dir = os.path.join('results', 'hsic')
        os.makedirs(hsic_dir, exist_ok=True)

        if config.hsic.split == "train":
            hsic_loader = train_loader
        elif config.hsic.split == "val":
            hsic_loader = val_loader
        elif config.hsic.split == "test":
            hsic_loader = test_loader
        else:
            raise ValueError(f"Unknown split: {config.hsic.split}. Please choose from 'train', 'val', or 'test'.")

        print(f">> Calculating HSIC(X,Z) and HSIC(Z,Y) on {config.hsic.split} set...")
        hsic_xz_lists, hsic_yz_lists = calculate_hsic(normalized_model, hsic_loader, config.hsic)
        print(">> done.")
        # print(f"len(hsic_xz_list): {len(hsic_xz_list)}, len(hsic_yz_list): {len(hsic_yz_list)}")
        combined_hsic_xz_list = sum(hsic_xz_lists, [])
        combined_hsic_yz_list = sum(hsic_yz_lists, [])

        # histogram paths
        xz_histogram_folder = os.path.join(hsic_dir, "histogram_xz")
        yz_histogram_folder = os.path.join(hsic_dir, "histogram_yz")
        os.makedirs(os.path.join(hsic_dir, "histogram_xz"), exist_ok=True)
        os.makedirs(os.path.join(hsic_dir, "histogram_yz"), exist_ok=True)
        xz_histogram_path = os.path.join(xz_histogram_folder, f"{config.training.wandb.run_name}.png")
        yz_histogram_path = os.path.join(yz_histogram_folder, f"{config.training.wandb.run_name}.png")

        # plot and save hsic_xz histogram
        try:
            plt.clf()
        except UnboundLocalError:
            pass
        plt = plot_histogram(combined_hsic_xz_list, bins=25, title='Histogram of HSIC(X,Z)', xlabel='HSIC(X,Z)', ylabel='Frequency', color='blue')
        plt.savefig(xz_histogram_path, bbox_inches='tight')
        if config.training.wandb.track:
            wandb.log({"hsic_xz_histogram": wandb.Image(xz_histogram_path)})

        # plot and save hsic_yz histogram
        plt.clf()
        plt = plot_histogram(combined_hsic_yz_list, bins=25, title='Histogram of HSIC(Z,Y)', xlabel='HSIC(Z,Y)', ylabel='Frequency', color='green')
        plt.savefig(yz_histogram_path, bbox_inches='tight')
        if config.training.wandb.track:
            wandb.log({"hsic_yz_histogram": wandb.Image(yz_histogram_path)})
        
        # save hsic lists to a pickle file
        hsic_dict = {
            "hsic_xz": combined_hsic_xz_list,
            "hsic_yz": combined_hsic_yz_list
        }
        hsic_filepath = f"results/hsic/{config.training.wandb.run_name}.pkl"
        with open(hsic_filepath, "wb") as f:
            pickle.dump(hsic_dict, f)

        # log average hsic values
        final_hsics = {}
        for i, (hsic_xz_list, hsic_yz_list) in enumerate(zip(hsic_xz_lists, hsic_yz_lists)):
            final_hsics[f"hsic_xz_{i}"] = sum(hsic_xz_list) / len(hsic_xz_list)
            final_hsics[f"hsic_yz_{i}"] = sum(hsic_yz_list) / len(hsic_yz_list)
            
        if config.training.wandb.track:
            # wandb.log({"hsic_xz": sum(hsic_xz_list) / len(hsic_xz_list), "hsic_yz": sum(hsic_yz_list) / len(hsic_yz_list)})
            wandb.log(final_hsics)
            wandb.save(hsic_filepath)

    if config.hcr.calculate:
        # make sure hcr directory exists
        hcr_dir = os.path.join('results', 'hcr')
        os.makedirs(hcr_dir, exist_ok=True)
        print(">> Calculating HCR...")

        # if model has a only_logits attribute, use it
        if hasattr(normalized_model, 'only_logits'):
            # change call method
            normalized_model.only_logits = True
        
        # Calculate HCR bounds
        hcr_run_name = config.training.wandb.run_name
        hcr, hcrmax = calculate_hcr(model=normalized_model,
                                    loader=val_loader,
                                    hcr_run_name=config.training.wandb.run_name,
                                    hcr_config=config.hcr)

        hcr_dict = {
            "hcr": hcr,
            "hcrmax": hcrmax
        }
        hcr_filepath = f"results/hcr/{config.training.wandb.run_name}.pkl"
        with open(hcr_filepath, "wb") as f:
            pickle.dump(hcr_dict, f)
        if config.training.wandb.track:
            wandb.save(hcr_filepath)

    if config.fisher.calculate:
        # make sure fisher directory exists
        fisher_dir = os.path.join('results', 'fisher')
        os.makedirs(fisher_dir, exist_ok=True)
        print("Fisher information calculation not ready yet")
        # fisher = diag_fisher(model, val_loader,
        #                     device=config.training.device,
        #                     )

    if config.auto_attack.calculate:
        # make sure auto_attack directory exists
        auto_attack_dir = os.path.join('results', 'auto_attack')
        os.makedirs(auto_attack_dir, exist_ok=True)

        # if model has a only_logits attribute, use it. This is for defenses that typically have multiple outputs in the forward pass, like BiDO or MID.
        if hasattr(normalized_model, 'only_logits'):
            # change call method
            normalized_model.only_logits = True
        
        if config.auto_attack.split == "train":
            print(">> Running AutoAttack on train set...")
            auto_attack_loader = train_loader
        elif config.auto_attack.split == "val":
            print(">> Running AutoAttack on val set...")
            auto_attack_loader = val_loader
        elif config.auto_attack.split == "test":
            print(">> Running AutoAttack on test set...")
            auto_attack_loader = test_loader
        else:
            raise ValueError(f"Unknown split: {config.auto_attack.split}. Please choose from 'train', 'val', or 'test'.")
        # Run AutoAttack
        auto_attack_out = auto_attack(normalized_model, auto_attack_loader, config.auto_attack, config)

    if config.umap.calculate:
        # make sure umap directory exists
        umap_dir = os.path.join('results', 'umap')
        os.makedirs(umap_dir, exist_ok=True)

        if config.umap.split == "train":
            print(">> Running UMAP on train set...")
            umap_loader = train_loader
        elif config.umap.split == "val":
            print(">> Running UMAP on val set...")
            umap_loader = val_loader
        elif config.umap.split == "test":
            print(">> Running UMAP on test set...")
            umap_loader = test_loader
        else:
            raise ValueError(f"Unknown split: {config.umap.split}. Please choose from 'train', 'val', or 'test'.")

        # plt.clf()
        plt = calculate_umap(normalized_model, umap_loader, config.umap, run_name=config.training.wandb.run_name, random_state=config.training.seed, device=config.training.device)
        plt.savefig(os.path.join(umap_dir, f"{config.training.wandb.run_name}.png"), format='png', dpi=1400)
        if config.training.wandb.track:
            wandb.log({"umap": wandb.Image(plt)})
        plt.clf()
    
    if config.cluster_metrics.calculate:
        # make sure cluster_metrics directory exists
        cluster_metrics_dir = "results/cluster_metrics"
        os.makedirs(cluster_metrics_dir, exist_ok=True)
        
        if config.cluster_metrics.split == "train":
            print(">> Running cluster metrics on train set...")
            cluster_loader = train_loader
        elif config.cluster_metrics.split == "val":
            print(">> Running cluster metrics on val set...")
            cluster_loader = val_loader
        elif config.cluster_metrics.split == "test":
            print(">> Running cluster metrics on test set...")
            cluster_loader = test_loader
        else:
            raise ValueError(f"Unknown split: {config.cluster_metrics.split}. Please choose from 'train', 'val', or 'test'.")
            
        cluster_metrics_dfs = run_cluster_metrics(normalized_model, cluster_loader, config.cluster_metrics)

        for normalization, df in cluster_metrics_dfs.items(): #! needs changing if this is ever used again
            run_name_split = config.training.wandb.run_name.split("__")
            if len(run_name_split) == 3: # the wandb run name is in our experiment format: model_name__custom_name__experiment_name
                model_name, custom_name, experiment_name = run_name_split
                custom_name += f"_{normalization}" # add normalization to custom name
                csv_name = f"{model_name}__{custom_name}__{experiment_name}"
            else:
                csv_name = f"{config.training.wandb.run_name}_{normalization}"

            output_file = os.path.join(cluster_metrics_dir, f"{csv_name}.csv")
            df.to_csv(output_file, index=False)
            print(f"{normalization} CSV saved to {output_file}")
            wandb.save(output_file)


    wandb.finish()
        
def calculate_top5_accuracy(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)

    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if hasattr(model, 'forward_only_logits'):
                # If the model has a forward_only_logits method, use it
                outputs = model.forward_only_logits(inputs)
            
            else:
                outputs = model(inputs)

            # Get top 5 predictions
            
            _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            
            # Check if target is in top 5 predictions
            correct_top5 += (top5_preds == targets.unsqueeze(1)).sum().item()
            total += targets.size(0)

    top5_accuracy = correct_top5 / total
    return top5_accuracy


def plot_histogram(data, bins, title, xlabel, ylabel, color):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', color=color, alpha=0.8)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.tight_layout()
    
    return plt
    
if __name__ == "__main__":
    evaluate_classifier()