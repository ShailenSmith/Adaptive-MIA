import os
import torch
from torch import nn
import wandb
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

from autoattack.autoattack import AutoAttack # using forked version

from utils.plotting import plot_tensor
from matplotlib import pyplot as plt

def auto_attack(model, loader, auto_attack_config, config, device='cuda'):
    """
    Runs AutoAttack on the given model and dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to attack.
        loader (torch.utils.data.DataLoader): loader for the dataset to evaluate.
        eps (float): The perturbation budget (epsilon) for the attack.
        device (str): The device to run the attack on ('cuda' or 'cpu').

    Returns:
        dict: A dictionary containing the clean accuracy and adversarial accuracy.
    """
    model.eval()
    model.to(device)

    # Prepare inputs and labels
    x_test = []
    y_test = []
    for i, (inputs, labels) in tqdm(enumerate(loader), desc="Loading data", total=len(loader)):
        x_test.append(inputs.to(device)) # if large enough, this needs to be on gpu for torch.cat to avoid CPU memory issues
        y_test.append(labels.to(device))
        if auto_attack_config.num_batches != "all":
            if i >= auto_attack_config.num_batches - 1:
                break

    print("concatenating x_test and y_test...")
    x_test = torch.cat(x_test, dim=0).to("cpu")
    y_test = torch.cat(y_test, dim=0).to("cpu")
    torch.cuda.empty_cache() # free up memory
    print("done.")

    assert (x_test.min() >= 0 and x_test.max() <= 1), f"Input data must be in the range [0, 1]. Got min: {x_test.min()}, max: {x_test.max()}"

    # Initialize AutoAttack
    adversary = AutoAttack(model, norm='Linf', eps=auto_attack_config.eps, version=auto_attack_config.version, device=device)
    torch.use_deterministic_algorithms(False) # there is an error with cuda cumsum (?) without this line

    attacks_to_run = []
    if auto_attack_config.run_apgd_ce:
        attacks_to_run.append("apgd-ce")
    if auto_attack_config.run_apgd_t:
        attacks_to_run.append("apgd-t")
    if auto_attack_config.run_fab_t:
        attacks_to_run.append("fab-t")
    if auto_attack_config.run_square:
        attacks_to_run.append("square")

    adversary.attacks_to_run = attacks_to_run

    # Run AutoAttack
    # get list of counts of each label in y_test
    label_counts = torch.zeros(config.dataset.n_classes).to(device)
    for label in y_test.unique():
        label_counts[label] = (y_test == label).sum()

    # set bs for batch size
    bs = loader.batch_size

    # start AutoAttack
    print("Starting AutoAttack...")
    if auto_attack_config.run_individual_attacks: # use run_standard_evaluation_individual
        with torch.no_grad():
            x_adv_dict = adversary.run_standard_evaluation_individual(x_test, y_test, bs=bs)

        # dictionaries will have one item for each attack in attacks_to_run
        total_misclassifications_dict = {}
        total_misclassifications_per_label_dict = {}
        total_loss_diffs_dict = {}
        total_loss_diffs_per_label_dict = {}
        is_misclassified_tensor_dict = {}
        robust_accuracy_check_dict = {}
        mean_loss_diff_dict = {}
        robust_accuracy_per_class_dict = {}
        mean_loss_diffs_per_class_dict = {}

        for attack in attacks_to_run:
            with torch.no_grad():
                x_adv, robust_accuracy = x_adv_dict[attack]
                assert x_adv.shape[0] == y_test.shape[0], f"x_adv is length {x_adv.shape[0]}, but y_test is length {y_test.shape[0]}"

                total_misclassifications = 0
                total_misclassifications_per_label = torch.zeros(config.dataset.n_classes).to(device)
                total_loss_diffs = 0
                total_loss_diffs_per_label = torch.zeros(config.dataset.n_classes).to(device)

                is_misclassified_tensor = torch.ones_like(y_test, dtype=torch.bool).to(device)

                for i in range(x_adv.shape[0] // bs):
                    start = bs*i
                    end = min(bs*(i+1), x_adv.shape[0])

                    normal_imgs = x_test[start:end].to(device)
                    adv_imgs = x_adv[start:end].to(device)
                    true_labels = y_test[start:end].to(device)

                    # normal_predicted_labels = model.predict(normal_imgs)
                    normal_outs = model(normal_imgs)
                    adv_outs = model(adv_imgs)

                    normal_losses = nn.CrossEntropyLoss(reduction='none')(normal_outs, true_labels)
                    adv_losses = nn.CrossEntropyLoss(reduction='none')(adv_outs, true_labels)

                    normal_predicted_labels = torch.argmax(normal_outs, dim=1)
                    adv_predicted_labels = torch.argmax(adv_outs, dim=1)

                    loss_diffs = adv_losses - normal_losses
                    total_loss_diffs += loss_diffs.sum().item()

                    is_misclassified = torch.logical_or(adv_predicted_labels != true_labels, normal_predicted_labels != true_labels)
                    total_misclassifications += is_misclassified.sum().item()
                    # save total misclassifications per label
                    for label in true_labels.unique():
                        total_misclassifications_per_label[label] += is_misclassified[true_labels == label].sum()
                        total_loss_diffs_per_label[label] += loss_diffs[true_labels == label].sum()

                    is_misclassified_tensor[start:end] = is_misclassified
                
                robust_accuracy_check = 1 - (total_misclassifications / x_adv.shape[0])
                mean_loss_diff = total_loss_diffs / x_adv.shape[0]
                print(f"Check robust accuracy for attack {attack}: ", robust_accuracy_check)
                print(f"Mean loss diff for attack {attack}: ", mean_loss_diff)

                robust_accuracy_per_class = 1 - (total_misclassifications_per_label / label_counts)
                mean_loss_diffs_per_class = total_loss_diffs_per_label / label_counts

                total_misclassifications_dict[attack] = total_misclassifications
                total_misclassifications_per_label_dict[attack] = total_misclassifications_per_label
                total_loss_diffs_dict[attack] = total_loss_diffs
                total_loss_diffs_per_label_dict[attack] = total_loss_diffs_per_label
                is_misclassified_tensor_dict[attack] = is_misclassified_tensor
                robust_accuracy_check_dict[attack] = robust_accuracy_check
                mean_loss_diff_dict[attack] = mean_loss_diff
                robust_accuracy_per_class_dict[attack] = robust_accuracy_per_class
                mean_loss_diffs_per_class_dict[attack] = mean_loss_diffs_per_class

            save_is_misclassified(is_misclassified_tensor, config.training.wandb.run_name, attack=attack)
            plot_adversarial_examples(x_adv, x_test, custom_name=attack, plot_name=config.training.wandb.run_name)
            wandb.log({
                f"{attack}__robust_accuracy_ORIG": robust_accuracy,
                f"{attack}__robust_accuracy_CHECK": robust_accuracy_check,
                f"{attack}__mean_loss_diff": mean_loss_diff,
            })
        
        # get which points got misclassified in any attack
        combined_misclassified = torch.zeros_like(list(is_misclassified_tensor_dict.values())[0], dtype=torch.bool)
        for attack in is_misclassified_tensor_dict.keys():
            combined_misclassified = torch.logical_or(combined_misclassified, is_misclassified_tensor_dict[attack])

        overall_robust_accuracy = 1 - combined_misclassified.float().mean()
        print(f"Overall robust accuracy: {overall_robust_accuracy:.4f}") #! note that is is calculated from the CHECK version, not ORIG
        wandb.log({
            "overall_robust_accuracy": overall_robust_accuracy.item(),
            "robust_accuracy": overall_robust_accuracy.item(),
        })
        
        auto_attack_df = pd.DataFrame({})
        for attack in attacks_to_run:
            auto_attack_df[f"{attack}__robust_accuracy"] = robust_accuracy_per_class_dict[attack].cpu().numpy()
            auto_attack_df[f"{attack}__total_misclassifications"] = total_misclassifications_per_label_dict[attack].cpu().numpy()
            auto_attack_df[f"{attack}__mean_loss_diff"] = mean_loss_diffs_per_class_dict[attack].cpu().numpy()
        auto_attack_df["label_counts"] = label_counts.cpu().numpy()
        # auto_attack_df["overall_robust_accuracy"] = overall_robust_accuracy.cpu().numpy()
        auto_attack_df.index.name = "class"

        save_auto_attack_df(auto_attack_df, config.training.wandb.run_name)

        wandb.log({
            "attacks": str(adversary.attacks_to_run),
        })

        return x_adv_dict

    else: # use run_standard_evaluation instead
        with torch.no_grad():
            x_adv, robust_accuracy = adversary.run_standard_evaluation(x_test, y_test, bs=bs)
        assert x_adv.shape[0] == y_test.shape[0], f"x_adv is length {x_adv.shape[0]}, but y_test is length {y_test.shape[0]}"

        # calculate robust accuracy, overall and per class
        total_misclassifications = 0
        total_misclassifications_per_label = torch.zeros(config.dataset.n_classes).to(device)

        total_loss_diffs = 0
        total_loss_diffs_per_label = torch.zeros(config.dataset.n_classes).to(device)

        is_misclassified_tensor = torch.ones_like(y_test, dtype=torch.bool).to(device)

        # run through the examples to find total misclassifications and loss diffs
        for i in range(x_adv.shape[0] // bs):
            with torch.no_grad():
                start = bs*i
                end = min(bs*(i+1), x_adv.shape[0])
                normal_imgs = x_test[start:end].to(device)
                adv_imgs = x_adv[start:end].to(device)
                true_labels = y_test[start:end].to(device)

                normal_outs = model(normal_imgs)
                adv_outs = model(adv_imgs)
                normal_losses = nn.CrossEntropyLoss(reduction='none')(normal_outs, true_labels)
                adv_losses = nn.CrossEntropyLoss(reduction='none')(adv_outs, true_labels)
                normal_predicted_labels = torch.argmax(normal_outs, dim=1)
                adv_predicted_labels = torch.argmax(adv_outs, dim=1)

                loss_diffs = adv_losses - normal_losses
                total_loss_diffs += loss_diffs.sum().item()

                is_misclassified = torch.logical_or(adv_predicted_labels != true_labels, normal_predicted_labels != true_labels)
                total_misclassifications += is_misclassified.sum().item()
                # save total misclassifications per label
                for label in true_labels.unique():
                    total_misclassifications_per_label[label] += is_misclassified[true_labels == label].sum()
                    total_loss_diffs_per_label[label] += loss_diffs[true_labels == label].sum()
                is_misclassified_tensor[start:end] = is_misclassified

        # calculate some metrics
        robust_accuracy_check = 1 - (total_misclassifications / x_adv.shape[0])
        mean_loss_diff = total_loss_diffs / x_adv.shape[0]
        print("Check robust accuracy: ", robust_accuracy_check)
        print("Mean loss diff: ", mean_loss_diff)

        # get robust accuracy and mean loss diff for each class
        robust_accuracy_per_class = 1 - (total_misclassifications_per_label / label_counts)
        mean_loss_diffs_per_class = total_loss_diffs_per_label / label_counts

        # log other important things
        save_is_misclassified(is_misclassified_tensor, config.training.wandb.run_name)
        plot_adversarial_examples(x_adv, x_test, custom_name=f"all{len(attacks_to_run)}", plot_name=config.training.wandb.run_name)
        wandb.log({
            "robust_accuracy": robust_accuracy,
            "mean_loss_diff": mean_loss_diff,
            "attacks": str(adversary.attacks_to_run),
        })

        # make pandas dataframe for per-class metrics
        auto_attack_df = pd.DataFrame({"robust_accuracy" : robust_accuracy_per_class.cpu().numpy(),
                        "total_misclassifications" : total_misclassifications_per_label.cpu().numpy(),
                        "mean_loss_diff" : mean_loss_diffs_per_class.cpu().numpy(),
                        "label_counts" : label_counts.cpu().numpy()})
        auto_attack_df.index.name = "class"

        save_auto_attack_df(auto_attack_df, config.training.wandb.run_name)

        return x_adv

def save_auto_attack_df(auto_attack_df, run_name):
    # save robust accuracies and loss diffs as csv
    os.makedirs(os.path.join("results", "auto_attack", "metrics"), exist_ok=True)
    auto_attack_df_path = os.path.join("results", "auto_attack", "metrics", f"{run_name}.csv")
    auto_attack_df.to_csv(auto_attack_df_path, index=True)
    print(f"Autoattack per-class metrics saved to {auto_attack_df_path}")
    wandb.save(auto_attack_df_path)

def save_is_misclassified(is_misclassified_tensor, run_name, attack=None):
    # save is_misclassified list
    os.makedirs(os.path.join("results", "auto_attack", "misclassified"), exist_ok=True)
    if attack is not None:
        os.makedirs(os.path.join("results", "auto_attack", "misclassified", attack), exist_ok=True)
        is_misclassified_path = os.path.join("results", "auto_attack", "misclassified", attack, f"{run_name}.pt")
    else:
        is_misclassified_path = os.path.join("results", "auto_attack", "misclassified", f"{run_name}.pt")

    torch.save(is_misclassified_tensor, is_misclassified_path)
    print(f"Misclassified list saved to {is_misclassified_path}")
    wandb.save(is_misclassified_path)

def plot_adversarial_examples(x_adv, x_test, custom_name, plot_name=""):
    # plot some examples
    try:
        for i in range(5):
            plt = plot_tensor(x_test[i].detach().cpu(), f"{plot_name}_original_example_{i}")
            wandb.log({f"{custom_name}__original_example_{i}": wandb.Image(plt)})
            plt.clf()
            plt = plot_tensor(x_adv[i].detach().cpu(), f"{plot_name}_adversarial_example_{i}")
            wandb.log({f"{custom_name}__adversarial_example_{i}": wandb.Image(plt)})
    except Exception as e:
        print(f"WARNING: Could not plot adversarial examples: {e}")

# make wrapper model that performs normalization as first part of forward pass
def normalize_model(model, config, mean, std, device): # do before loading model weights

    class NormalizationWrapper(model.__class__):
        def __init__(self, mean, std, device):
            super(NormalizationWrapper, self).__init__(config)
            self.mean = torch.Tensor([0.5, 0.5, 0.5]).float().view(3, 1, 1).to(device)
            self.std = torch.Tensor([0.5, 0.5, 0.5]).float().view(3, 1, 1).to(device)

        def forward(self, x):
            x = (x - self.mean) / self.std
            x = super(NormalizationWrapper, self).forward(x)
            return x

    normalized_model = NormalizationWrapper(mean, std, device)
    return normalized_model
