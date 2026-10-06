import numpy as np
import torch

import umap
import matplotlib.pyplot as plt

def calculate_umap(model, loader, umap_config, run_name, random_state= 42, device='cuda'):
    """
    Runs UMAP on the given model and dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to visualize.
        loader (torch.utils.data.DataLoader): loader for the dataset to visualize.
        device (str): The device to run the visualization on ('cuda' or 'cpu').

    Returns:
        None
    """
    model.eval()
    model.to(device)

    # Prepare inputs and labels
    xs = []
    ys = []
    outs = []
    for i, (inputs, labels) in enumerate(loader):
        # xs.append(inputs.to(device))
        ys.append(labels[labels%265<=49].detach().cpu().numpy())
        outs.append(model.embed_img(inputs[labels%265<=49].to(device)).detach().cpu().numpy())
        # if umap_config.num_batches != "all":
        #     if i >= umap_config.num_batches - 1:
        #         break

    # xs = torch.cat(xs, dim=0)
    ys = np.concatenate(ys, axis=0)
    outs = np.concatenate(outs, axis=0)

    umapper = umap.UMAP(n_neighbors=umap_config.n_neighbors, min_dist=umap_config.min_dist,
                        n_components=umap_config.n_components, metric=umap_config.metric, random_state=random_state)
    embedding = umapper.fit_transform(outs)

    # For categorical labels
    plt.figure(figsize=(6, 4))

    # Convert labels to categorical if they're not already
    unique_ys = np.unique(ys)
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_ys)))

    # Plot each class with a different color
    np.random.seed(random_state)
    np.random.shuffle(unique_ys)

    for i, y in enumerate(unique_ys):
        indices = np.where(ys == y)[0]
        plt.scatter(embedding[indices, 0], embedding[indices, 1], 
                    color=colors[i], label=f'Class {y}', s=2, alpha=0.8, edgecolor='none')

    plt.title(f'UMAP -- {run_name}')

    return plt



