import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import wandb
from sklearn.metrics import silhouette_samples

def run_cluster_metrics(model, loader, cluster_metrics_config, device='cuda'):
    model.eval()
    model.to(device)

    # Prepare inputs and labels
    xs = []
    all_labels = []
    outs = []
    num_batches = cluster_metrics_config.num_batches
    num_iter = len(loader) if num_batches == "all" else min(num_batches, len(loader))
    for i, (inputs, labels) in tqdm(enumerate(loader), desc="Collecting embeddings...", total=num_iter):
        all_labels.append(labels.detach()) #.cpu().numpy())
        outs.append(model.embed_img(inputs.to(device)).detach()) #.cpu().numpy())

        if cluster_metrics_config.num_batches != "all":
            if i >= num_batches:
                break

    all_labels = torch.cat(all_labels, dim=0)
    outs = torch.cat(outs, dim=0)

    # calculate L2 distances between all distinct pairs of points in outs, plus some normalizations
    raw_dists = torch.cdist(outs, outs, p=2)
    max_normalized_dists = raw_dists / torch.max(raw_dists)
    zscore_normalized_dists = (raw_dists - torch.mean(raw_dists)) / torch.std(raw_dists)

    raw_cluster_metrics_df = calculate_cluster_metrics(raw_dists, all_labels, device, normalization="raw")
    # print(f'Raw cluster metrics: {raw_cluster_metrics}')
    max_normalized_cluster_metrics_df = calculate_cluster_metrics(max_normalized_dists, all_labels, device, normalization="max")
    # print(f'Max normalized cluster metrics: {max_normalized_cluster_metrics}')
    zscore_normalized_cluster_metrics_df = calculate_cluster_metrics(zscore_normalized_dists, all_labels, device, normalization="zscore")
    # print(f'Z-score normalized cluster metrics: {zscore_normalized_cluster_metrics}')


    cluster_metrics_dfs = {}
    cluster_metrics_dfs['raw'] = raw_cluster_metrics_df
    cluster_metrics_dfs['max'] = max_normalized_cluster_metrics_df
    cluster_metrics_dfs['zscore'] = zscore_normalized_cluster_metrics_df

    # calculate silhouette score
    if cluster_metrics_config.calculate_silhouette:
        silhouette_vals = silhouette_samples(outs.cpu().numpy(), all_labels.cpu().numpy())
        wandb.log({"silhouette_score": np.mean(silhouette_vals)})
        silhouette_df = pd.DataFrame(silhouette_vals, columns=["silhouette_score"])

    return cluster_metrics_dfs


def calculate_cluster_metrics(dists, all_labels, device, normalization="raw"):
    # calculate mean intra-class distance, not including self-distances
    intra_class_dists = []
    for i in torch.unique(all_labels):
        mask = all_labels == i
        class_dists = dists[mask][:, mask].clone()
        diagonal_mask = torch.eye(class_dists.shape[0], dtype=bool).to(device)
        class_dists[diagonal_mask] = float('inf')  # Set self-distances to infinity
        intra_class_dists.append(torch.mean(class_dists[class_dists != float('inf')]))
    mean_intra_class_dist = torch.mean(torch.stack(intra_class_dists))
    # print(f'Mean intra-class distance: {mean_intra_class_dist}')

    # calculate mean inter-class distance
    inter_class_dists = []
    for i in torch.unique(all_labels):
        mask = all_labels == i
        inter_class_dists.append(torch.mean(dists[~mask][:, mask]))
    mean_inter_class_dist = torch.mean(torch.stack(inter_class_dists))
    # print(f'Mean inter-class distance: {mean_inter_class_dist}')

    classes = []
    
    intra_class_nn_dists = {}
    intra_class_nn_dists_means = []
    intra_class_nn_dists_medians = []
    intra_class_nn_dists_fqrts = []
    intra_class_nn_dists_tqrts = []
    intra_class_nn_dists_mins = []
    intra_class_nn_dists_maxes = []

    inter_class_nn_dists = {}
    inter_class_nn_dists_means = []
    inter_class_nn_dists_medians = []
    inter_class_nn_dists_fqrts = []
    inter_class_nn_dists_tqrts = []
    inter_class_nn_dists_mins = []
    inter_class_nn_dists_maxes = []

    intra_inter_nn_min_ratios = []
    intra_inter_nn_mean_ratios = []
    intra_inter_nn_median_ratios = []
    qtensor = torch.tensor([0.25, 0.5, 0.75]).to(device)

    # inter_class_nn_dists = []
    labels_to_ignore = []
    for i in tqdm(torch.unique(all_labels), desc=f"{normalization}: Calculating intra and inter class nearest neighbor distances..."):
        mask = all_labels == i
        i = i.item() # convert to int
        classes.append(i)

        if mask.sum() < 2: # below will fail if there is only one sample in the class
            intra_class_nn_dists[i] = torch.empty(2, 2).to(device)
            labels_to_ignore.append(i)
            print(f'Class {i} has only one sample, skipping intra-class nearest neighbor distance calculation.')
        else:
            intra_class_nn_dists[i] = dists[mask][:, mask].topk(2, dim=1, largest=False)[0][:, 1] # take 2nd nearest neighbor to avoid self-distance

        # calculate means, medians, mins, maxes, and quartiles for each class
        intra_class_nn_dists_means.append(torch.mean(intra_class_nn_dists[i]).item())
        intra_class_nn_dists_mins.append(torch.min(intra_class_nn_dists[i]).item())
        intra_class_nn_dists_maxes.append(torch.max(intra_class_nn_dists[i]).item())
        quartiles = torch.quantile(intra_class_nn_dists[i], qtensor) # first, second, and third quartiles
        intra_class_nn_dists_fqrts.append(quartiles[0].item())
        intra_class_nn_dists_medians.append(quartiles[1].item())
        intra_class_nn_dists_tqrts.append(quartiles[2].item())

        inter_class_nn_dists[i] = dists[mask][:, ~mask].topk(1, dim=1, largest=False)[0][:, 0] # no need to avoid self-distance here
        # calculate means, medians, mins, maxes, and quartiles for each class
        inter_class_nn_dists_means.append(torch.mean(inter_class_nn_dists[i]).item())
        inter_class_nn_dists_mins.append(torch.min(inter_class_nn_dists[i]).item())
        inter_class_nn_dists_maxes.append(torch.max(inter_class_nn_dists[i]).item())
        quartiles = torch.quantile(inter_class_nn_dists[i], qtensor) # first, second, and third quartiles
        inter_class_nn_dists_fqrts.append(quartiles[0].item())
        inter_class_nn_dists_medians.append(quartiles[1].item())
        inter_class_nn_dists_tqrts.append(quartiles[2].item())

        # calculate ratios of intra-class to inter-class nearest neighbor distances
        intra_inter_nn_min_ratios.append(intra_class_nn_dists_mins[-1] / inter_class_nn_dists_mins[-1])
        intra_inter_nn_median_ratios.append(intra_class_nn_dists_medians[-1] / inter_class_nn_dists_medians[-1])
        intra_inter_nn_mean_ratios.append(intra_class_nn_dists_means[-1] / inter_class_nn_dists_means[-1])



    # Create a DataFrame from the lists
    data = {
        "class": [x for i, x in enumerate(classes) if i not in labels_to_ignore],
        "intra_class_nn_dists_mean": [x for i, x in enumerate(intra_class_nn_dists_means) if i not in labels_to_ignore],
        "intra_class_nn_dists_min": [x for i, x in enumerate(intra_class_nn_dists_mins) if i not in labels_to_ignore],
        "intra_class_nn_dists_fqrt": [x for i, x in enumerate(intra_class_nn_dists_fqrts) if i not in labels_to_ignore],
        "intra_class_nn_dists_median": [x for i, x in enumerate(intra_class_nn_dists_medians) if i not in labels_to_ignore],
        "intra_class_nn_dists_tqrt": [x for i, x in enumerate(intra_class_nn_dists_tqrts) if i not in labels_to_ignore],
        "intra_class_nn_dists_max": [x for i, x in enumerate(intra_class_nn_dists_maxes) if i not in labels_to_ignore],
        "inter_class_nn_dists_mean": [x for i, x in enumerate(inter_class_nn_dists_means) if i not in labels_to_ignore],
        "inter_class_nn_dists_min": [x for i, x in enumerate(inter_class_nn_dists_mins) if i not in labels_to_ignore],
        "inter_class_nn_dists_fqrt": [x for i, x in enumerate(inter_class_nn_dists_fqrts) if i not in labels_to_ignore],
        "inter_class_nn_dists_median": [x for i, x in enumerate(inter_class_nn_dists_medians) if i not in labels_to_ignore],
        "inter_class_nn_dists_tqrt": [x for i, x in enumerate(inter_class_nn_dists_tqrts) if i not in labels_to_ignore],
        "inter_class_nn_dists_max": [x for i, x in enumerate(inter_class_nn_dists_maxes) if i not in labels_to_ignore],
    }
    df = pd.DataFrame(data)

    # summary logging
    intra_inter_nn_min_ratios = [x for i, x in enumerate(intra_inter_nn_min_ratios) if i not in labels_to_ignore]
    intra_inter_nn_mean_ratios = [x for i, x in enumerate(intra_inter_nn_mean_ratios) if i not in labels_to_ignore]
    intra_inter_nn_median_ratios = [x for i, x in enumerate(intra_inter_nn_median_ratios) if i not in labels_to_ignore]

    min_key_prefix = f'{normalization}_minratio_'
    min_ratio_stats = pd.DataFrame(intra_inter_nn_min_ratios).describe()[0].rename(
        {"25%": f"{min_key_prefix}fqrt", "50%": f"{min_key_prefix}median", "75%": f"{min_key_prefix}tqrt",
        "mean": f"{min_key_prefix}mean", "std": f"{min_key_prefix}std", "count": f"{min_key_prefix}count",
        "min": f"{min_key_prefix}min", "max": f"{min_key_prefix}max"}).to_dict()
    wandb.log(min_ratio_stats)

    mean_key_prefix = f'{normalization}_meanratio_'
    mean_ratio_stats = pd.DataFrame(intra_inter_nn_mean_ratios).describe()[0].rename(
        {"25%": f"{mean_key_prefix}fqrt", "50%": f"{mean_key_prefix}median", "75%": f"{mean_key_prefix}tqrt",
        "mean": f"{mean_key_prefix}mean", "std": f"{mean_key_prefix}std", "count": f"{mean_key_prefix}count",
        "min": f"{mean_key_prefix}min", "max": f"{mean_key_prefix}max"}).to_dict()
    wandb.log(mean_ratio_stats)

    median_key_prefix = f'{normalization}_medianratio_'
    median_ratio_stats = pd.DataFrame(intra_inter_nn_median_ratios).describe()[0].rename(
        {"25%": f"{median_key_prefix}fqrt", "50%": f"{median_key_prefix}median", "75%": f"{median_key_prefix}tqrt",
        "mean": f"{median_key_prefix}mean", "std": f"{median_key_prefix}std", "count": f"{median_key_prefix}count",
        "min": f"{median_key_prefix}min", "max": f"{median_key_prefix}max"}).to_dict()
    wandb.log(median_ratio_stats)

    return df
