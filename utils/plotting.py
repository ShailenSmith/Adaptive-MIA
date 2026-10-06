import matplotlib.pyplot as plt
import torch
import os

def plot_tensor(tensor, save_name):
    if len(tensor.shape) == 3: # assume this is of form (C, H, W)
        arr = tensor.numpy().transpose(1,2,0)
    else:
        arr = tensor.numpy()
    # Create figure and axis to have fine-grained control over padding and DPI
    fig, ax = plt.subplots()
    ax.imshow(arr)
    ax.axis('off')

    # Ensure output directory exists
    out_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"{save_name}.png")

    # Save with no padding or margins to avoid white border
    fig.savefig(plot_path, bbox_inches='tight', pad_inches=0)
    return plt


def plot_lasso_mask(drop_layer_model, plot_name):
    w_first = drop_layer_model.mask_layer.weight.data
                    
    n_channels, x_dim, y_dim = drop_layer_model.config.dataset.input_size
    if n_channels == 3:
        w_norms = torch.linalg.norm(w_first, dim=0)
    else: # n_channels == 1
        w_norms = w_first.abs() # does the same thing as norm of dim=0 when n_channels is 1, but this is more readable
        
    w_norms = w_norms.reshape((1, x_dim, y_dim)) # (x_dim, y_dim) -> (1, x_dim, y_dim)

    plt = plot_tensor(w_norms.cpu(), plot_name)
    
    return plt
    
def plot_masked_image(drop_layer_model, tensor_images, model_id, multiply_by):
    lasso_mask = drop_layer_model.mask_layer
    
    
    
    tensor_images = torch.stack(tensor_images).to(drop_layer_model.device)
    
    tensor_images = (tensor_images + 1) / 2  # Transform from [-1, 1] range to [0, 1] range
    
    masked_images = lasso_mask(tensor_images)
    
    for i, image in enumerate(masked_images):
        
        plot_tensor(image.detach().cpu() * multiply_by, f"{model_id}_masked_image_x{multiply_by}_{i}")
        
    for i, image in enumerate(tensor_images):

        plot_tensor(image.detach().cpu(), f"{model_id}_original_image_{i}")
