import torch
import torch.nn as nn
import torchvision
import wandb
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf

import os, shutil, time
from tqdm import tqdm
import copy

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from data_processing.data_augmentation import get_transforms
import data_processing

from classifiers.abstract_classifier import AbstractClassifier
from classifiers.get_model import get_model
from utils import wandb_helpers
from utils.plotting import plot_tensor


def apply_drop_layer_defense(config, model:AbstractClassifier):
    
    class DropLayerClassifier(model.__class__):
        
        def __init__(self, config):
            super(DropLayerClassifier, self).__init__(config)
            self.mask_layer = self.init_mask_layer()
            self.epoch = 0

            

            try:
                self.use_frozen_custom_mask = self.config.defense.use_frozen_custom_mask
            except Exception as e:
                print("Warning: config.defense.use_frozen_custom_mask not in struct. Default setting to False.")
                self.use_frozen_custom_mask = False
            
            # if using adaptive group lasso, load pre-adapted defense layer weights
            try:
                self.adaptive = config.defense.lasso.adaptive
            except Exception as e:
                print("Warning: config.defense.adaptive not in struct. Default setting to False.")
                self.adaptive = False

            try:
                self.mask_method = config.defense.mask_method
            except Exception as e:
                self.mask_method = input("Warning: config.defense.mask_method not in struct. Please set mask_method here: ")

            if self.adaptive:
                pre_adapted_mask_layer = self.get_pre_adapted_mask_layer() # note this is a Tensor, as opposed to self.mask_layer which is an ElementwiseLinear module
                assert pre_adapted_mask_layer.shape == self.get_mask().data.shape, "pre-adapted mask layer and current mask layer are different shapes"
                self.pre_adapted_mask_layer_norms = torch.linalg.norm(pre_adapted_mask_layer, dim=0).to(self.device) # we only need the norms

                # # if a pre-adapted pixel norm is 0, make the same pixel norm 0 in our current model
                # mask = self.get_mask()
                # mask.data[pre_adapted_mask_layer == 0] = 0
                # self.set_mask(mask)

            # some configuration stuff to make thresholding work
            if self.config.defense.apply_threshold:
                try:
                    self.initial_threshold = self.config.defense.lasso.initial_threshold
                    self.change_threshold_epoch = self.config.defense.lasso.change_threshold_epoch
                except Exception as e:
                    print("Warning: initial_threshold or change_threshold_epoch not found in struct, default setting to 1e-6 and 10 respectively")
                    self.initial_threshold = 1e-6
                    self.change_threshold_epoch = 10
            
                self.threshold = self.initial_threshold
            
            # mask configuration stuff

        def compute_masked_dataset(self, train_loader):

            if os.path.isdir(self.directory):
                shutil.rmtree(self.directory)
            
            os.makedirs(self.directory, exist_ok=True)
            
            max_string_length = len(str(self.config.dataset.n_classes)) + 1
            folder_names = []
            for i in range(self.config.dataset.n_classes):
                
                string_length = len(str(i))
                if string_length < max_string_length:
                    folder_name = str(i).zfill(max_string_length)
                
                os.makedirs(os.path.join(self.directory, folder_name), exist_ok=True)
                
                folder_names.append(folder_name)

            self.to(self.device)
            for batch_idx, (x, y) in tqdm(enumerate(train_loader), desc="Generating drop-layer dataset", total=len(train_loader)):
                x, y = x.to(self.device), y.to(self.device)

                masked_x = self.mask_layer(x)

                for i in range(masked_x.size(0)):
                    img = masked_x[i].detach().cpu()
                    label = y[i].item()
                    folder_name = folder_names[label]
                    
                    # save_path = f"defenses/non_robust_data/{self.config.dataset.dataset}/{folder_name}/{batch_idx}_{i}.png"
                    save_path = os.path.join(self.directory, folder_name, f"{batch_idx}_{i}.png")
                    img = (img + 1) / 2 # Convert from [-1, 1] to [0, 1]
                    torchvision.utils.save_image(img, save_path)


        def train_model(self, train_loader, val_loader):
            
            freeze_mask = False
            if self.config.defense.load_only_mask:
                self.feature_extractor, self.classification_layer = self.init_model() # replace the loaded model with an unloaded model
                freeze_mask = True
            elif self.use_frozen_custom_mask:
                custom_mask = torch.load(self.use_frozen_custom_mask) # load custom mask from path name given
                self.set_mask(custom_mask)
                freeze_mask = True

            if freeze_mask:
                for param in self.mask_layer.parameters(): # freeze the mask layer during training
                    param.requires_grad = False

            if torch.cuda.device_count() > 1:
                self.mask_layer = nn.DataParallel(self.mask_layer) # self.feature_extractor and self.classification_layer will be put on DataParallel in super train_model call, so we just need to put the mask_layer on DataParallel here


            # Don't make the dataaugmentations for assembling the new drop layer dataset.
            base_augmentations = get_transforms(self.config, train=False)
            self.full_augmentations = train_loader.dataset.subset.dataset.transform

            if self.mask_method == "masked_dataset":

                # Include only base augmentations in train loader
                train_loader.dataset.subset.dataset.transform = base_augmentations

                # set directory either for saving/loading the masked dataset if recompute=True, or just loading it if recompute=False
                self.directory = os.path.join("defenses", "drop_layer_data", self.config.defense.masked_dataset.directory_name)
                
                # always recompute the dataset, to make sure self.mask_layer is the same as the mask used to generate the dataset
                self.compute_masked_dataset(train_loader)
                print("sleeping for 10 seconds to make sure the files are written...")
                for i in range(10):
                    print(f"Sleeping for {10-i} seconds...")
                    time.sleep(1)
                    
                # load in masked dataset with augmentations
                augmentations = self.full_augmentations # add augmentations back if there were any
                masked_dataset = datasets.ImageFolder(root=self.directory, 
                                                        transform=augmentations)
                
                # make dataloader and train
                masked_dataset_loader = DataLoader(masked_dataset,
                                            batch_size=train_loader.batch_size, 
                                            shuffle=True,
                                            num_workers=config.training.dataloader_num_workers,
                                            pin_memory=True,
                                            drop_last=True)
                super(DropLayerClassifier, self).train_model(masked_dataset_loader, val_loader)

            elif self.mask_method == "post_mask_aug":
                image_height = config.dataset.input_size[1]
                image_width  = config.dataset.input_size[2]
                image_size = (image_height, image_width)
                train_loader.dataset.subset.dataset.transform = transforms.Compose([
                    transforms.Resize(image_size, antialias=True),
                    transforms.ToTensor()
                ])

                super(DropLayerClassifier, self).train_model(train_loader, val_loader)

            elif self.mask_method == "mask_as_transform":

                image_height = config.dataset.input_size[1]
                image_width  = config.dataset.input_size[2]
                image_size = (image_height, image_width)
                    
                
                mask_layer_for_transform = copy.deepcopy(self.mask_layer).to("cpu")
                mask_transforms = transforms.Compose([
                    transforms.Resize(image_size, antialias=True),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    mask_layer_for_transform,
                    transforms.Lambda(lambda x: (x + 1) / 2), # un-normalize the image after applying mask
                    transforms.ToPILImage(),
                ])
                train_loader.dataset.subset.dataset.transform = transforms.Compose([
                    mask_transforms,
                    self.full_augmentations,
                ])

                super(DropLayerClassifier, self).train_model(train_loader, val_loader)

            elif self.mask_method == "mask_layer":
                super(DropLayerClassifier, self).train_model(train_loader, val_loader)

        def init_mask_layer(self):
            if self.config.model.flatten:
                in_features = (self.config.dataset.input_size[1] * self.config.dataset.input_size[2] * self.config.dataset.input_size[0],)
            else:
                in_features = (self.config.dataset.input_size[0], self.config.dataset.input_size[1], self.config.dataset.input_size[2])

            mask_layer = ElementwiseLinear(in_features, w_init=self.config.defense.mask_init)
            return mask_layer

        def get_mask(self): # regardless of if DataParallel or not
            if isinstance(self.mask_layer, nn.DataParallel):
                return self.mask_layer.module.weight
            else:
                return self.mask_layer.weight

        def set_mask(self, new_mask: torch.Tensor): # regardless of if DataParallel or not
            if isinstance(self.mask_layer, nn.DataParallel):
                self.mask_layer.module.weight.data = new_mask
            else:
                self.mask_layer.weight.data = new_mask

        def train_one_epoch(self, train_loader):
            self.epoch += 1
            self.train()
            total_loss = 0
            loss_calculated = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self(data)

                loss = self.get_loss(output, target)

                total_loss += loss.item()          

                loss_calculated += len(data)

                if self.config.training.verbose == 2:
                    print("loss: ", loss.item())

                loss.backward()
                self.optimizer.step()
                self.train_step += 1

                if self.config.defense.mask_method == "mask_layer":
                    if self.config.defense.apply_threshold:
                        self.apply_threshold()
                track_features = self.config.defense.lasso.track_features
                if track_features:
                    if isinstance(track_features, ListConfig):
                        feature_norms = self.get_feature_norms()
                        for idx, feature_norm in zip(track_features, feature_norms):
                            wandb.log({f"feature_{idx}" : feature_norm.item(), "train_step": self.train_step})
                    
                    if hasattr(self, 'n_features_remaining'):
                        n_features = self.n_features_remaining
                    else:
                        n_channels, _, _ = self.config.dataset.input_size
                        n_features = (self.get_mask()!=0).sum()/n_channels # should be number of features remaining
                    
                    wandb.log({"n_features" : n_features, "train_step" : self.train_step})

            train_loss = total_loss / loss_calculated

            # post-epoch logging stuff
            if self.epoch % self.config.defense.save_mask_layer_freq == 0:
                # save mask layer plot
                if self.config.defense.plot_mask:
                    w_first = self.get_mask().data
                    
                    n_channels, x_dim, y_dim = self.config.dataset.input_size
                    if n_channels == 3:
                        w_norms = torch.linalg.norm(w_first, dim=0)
                    else: # n_channels == 1
                        w_norms = w_first.abs() # does the same thing as norm of dim=0 when n_channels is 1, but this is more readable
                        
                    w_norms = w_norms.reshape((1, x_dim, y_dim)) # (x_dim, y_dim) -> (1, x_dim, y_dim)

                    try:
                        plt = plot_tensor(w_norms.cpu(), self.save_as)
                        if self.config.training.wandb.track:
                            wandb.log({"defense_mask" : plt, "train_step": self.train_step, "epoch": self.epoch})
                    except Exception as e:
                        print(f"WARNING: Could not plot defense mask: {e}")

            if self.config.defense.apply_threshold:
                if self.epoch == self.change_threshold_epoch: # change from 'initial_threshold' to 'threshold' when self.epoch == change_threshold_epoch
                    self.threshold = self.config.defense.lasso.threshold

            return train_loss

        def forward(self, x):
            if self.mask_method == "masked_dataset":
                if self.training: # method == "masked_dataset" uses a train loader whose images are already masked, so no need to apply a mask in forward
                    x = super(DropLayerClassifier, self).forward(x)
                else: # the val and test loaders are assumed to have unmasked images, so we apply mask_layer in the forward
                    x = self.mask_layer(x)
                    x = super(DropLayerClassifier, self).forward(x)
                
            elif self.mask_method == "post_mask_aug": #! expects that train_loader dataset does only ToTensor(), no normalization
                raise NotImplementedError("work in progress")
                x = self.mask_layer(x) # pass through the mask layer first
                if self.training:
                    # make into PIL image
                    for img in x:
                        img = TF.to_pil_image(img)
                        img = self.full_augmentations(img) # do augmentations

                x = super(DropLayerClassifier, self).forward(x)
            
            elif self.mask_method == "mask_as_transform":
                if self.training: # method == "mask_as_transform" uses a train loader with the mask_layer as one of its transforms, so no need to apply a mask in forward
                    x = super(DropLayerClassifier, self).forward(x)
                else: # the val and test loaders are assumed not to have mask_layer in its transforms, so we apply the mask in forward
                    x = self.mask_layer(x)
                    x = super(DropLayerClassifier, self).forward(x)

            elif self.mask_method == "mask_layer": # just apply the mask layer before passing to the rest of the model.
                x = self.mask_layer(x)
                x = super(DropLayerClassifier, self).forward(x)
            
            return x
        
        def evaluate(self, loader):
            #! changes to the dataset objects elsewhere will likely necessitate changes here
            no_aug_loader = copy.deepcopy(loader) 
            if self.config.dataset.augment_data: # for the train_loader we need to remove augmentations before passing to evaluate()
                if isinstance(loader.dataset, data_processing.datasets.AttackDataset):
                    base_augmentations = get_transforms(self.config, train=False)
                    no_aug_loader.dataset.subset.dataset.transform = base_augmentations
                elif isinstance(loader.dataset, torchvision.datasets.folder.ImageFolder): # should be a train loader formed from mask_method=MaskedDataset, should also remove augmentations from train loader
                    base_augmentations = get_transforms(self.config, train=False)
                    no_aug_loader.dataset.transform = base_augmentations
                elif isinstance(loader.dataset, torch.utils.data.dataset.Subset):
                    pass # val_loader and test_loader are of this form, no change necessary with those
            return super(DropLayerClassifier, self).evaluate(no_aug_loader)

        def embed_img(self, x):
            x = self.mask_layer(x)
            x = self.feature_extractor(x)
            return x

        def get_hiddens(self, x):
            embeddings_list = []
            arch = ""

            if type(self.feature_extractor) == nn.DataParallel:
                if type(self.feature_extractor.module) == torchvision.models.resnet.ResNet:
                    arch = 'resnet'
                else:
                    arch = 'unknown' # no explicit intermediate layers have been determined for BiDO calculation, so will default to just using the output of the feature extractor
            elif type(self.feature_extractor) == torchvision.models.resnet.ResNet:
                arch = 'resnet'
            else:
                arch = 'unknown'

            x = self.mask_layer(x)

            if arch == 'resnet': # use intermediate outputs from each of the four major ResNet blocks
                x = self.feature_extractor.conv1(x)
                x = self.feature_extractor.bn1(x)
                x = self.feature_extractor.relu(x)
                x = self.feature_extractor.maxpool(x)

                hidden1 = self.feature_extractor.layer1(x)
                embeddings_list.append(hidden1)

                hidden2 = self.feature_extractor.layer2(hidden1)
                embeddings_list.append(hidden2)

                hidden3 = self.feature_extractor.layer3(hidden2)
                embeddings_list.append(hidden3)

                hidden4 = self.feature_extractor.layer4(hidden3)
                embeddings_list.append(hidden4)

                z = self.feature_extractor.avgpool(hidden4)
                z = torch.flatten(z, 1)
                logits = self.classification_layer(z)
            
            else: # just use the final output before classification layer
                z = self.feature_extractor(x)
                embeddings_list.append(z)
                logits = self.classification_layer(z)

            return embeddings_list

        def get_outputs(self, x):
            return self.forward(x)

        def get_loss(self, output, target):
            loss = super(DropLayerClassifier, self).get_loss(output, target)
            if self.config.defense.penalty: # add penalty term to loss
                lasso_pen, ridge_pen = self.get_penalties()
                loss = loss + (self.config.defense.lasso.lambda_ * lasso_pen) + (self.config.defense.lasso.ridge_lambda * ridge_pen) 
            return loss

        def get_penalties(self): # Penalty on one-dimensional weights
            w_first = self.get_mask()
            w_norms = torch.linalg.norm(w_first, dim=0) # takes L2 norm over n_channels. Performs abs() when n_channels=1, combines RGB values of pixels (group lasso) when n_channels=3
            if self.adaptive: # Calculate GL+AGL (Dinh and Ho, 2020)
                eps = torch.tensor(1e-8) # to avoid division by zero
                adaptive_gamma = torch.tensor(self.config.defense.lasso.adaptive_gamma)
                adaptive_w_norms = torch.div(w_norms, torch.pow(self.pre_adapted_mask_layer_norms, adaptive_gamma)+eps)
            lasso_penalty = adaptive_w_norms.sum() if self.adaptive else w_norms.sum()
            ridge_penalty = torch.pow(torch.linalg.norm(w_norms), 2)
            return lasso_penalty, ridge_penalty

        def apply_threshold(self):
            current_w_first = self.get_mask().data # (n_channels, x_dim, y_dim)
            current_w_norms = torch.linalg.norm(current_w_first, dim=0) # (x_dim, y_dim)
            below_threshold = current_w_norms <= self.threshold  # (x_dim, y_dim)
            new_w_first = current_w_first * ~below_threshold # (n_channels, x_dim, y_dim) x (x_dim, y_dim) = (n_channels, x_dim, y_dim)
            self.set_mask(new_w_first)
            self.n_features_remaining = below_threshold.numel() - below_threshold.sum()

        def get_feature_norms(self):
            w_first = self.get_mask().data
            w_norms = torch.linalg.norm(w_first, dim=0) # takes L2 norm over n_channels, shape is (x_dim, y_dim)
            feature_idxs = OmegaConf.to_object(self.config.defense.lasso.track_features)
            if isinstance(feature_idxs[0], list): # like drop_layer, one weight for each feature index, norm is just abs. value
                w_features = [w_norms[*feature_idx] for feature_idx in feature_idxs]
                return [torch.abs(w_feature) for w_feature in w_features]
            elif isinstance(feature_idxs[0], int):
                if len(w_first.shape) == 1: # just one weight for each feature index, norm is just abs. value
                    w_features = [w_norms[feature_idx] for feature_idx in feature_idxs]
                    return [torch.abs(w_feature) for w_feature in w_features]
            else:
                assert 0 == 1, f"feature_idxs[0] is type {type(feature_idxs[0])}, not int or list"

        def get_pre_adapted_mask_layer(self):
            # loads the whole pre-adapted model, and keeps only the mask layer weights.
            pre_adapted_config, _ = wandb_helpers.get_config(
                entity=self.config.defense.lasso.pre_adapted_entity,
                project=self.config.defense.lasso.pre_adapted_project,
                run_id=self.config.defense.lasso.pre_adapted_run_id,
            )
            pre_adapted_weights_path = wandb_helpers.get_weights(
                entity=self.config.defense.lasso.pre_adapted_entity,
                project=self.config.defense.lasso.pre_adapted_project,
                run_id=self.config.defense.lasso.pre_adapted_run_id,
            )
            pre_adapted_model = get_model(pre_adapted_config)
            pre_adapted_model.mask_layer = self.init_mask_layer() # placeholder mask layer needed to load in the pre-adapted mask layer

            # Load model weights
            pre_adapted_model.load_model(pre_adapted_weights_path)
            return pre_adapted_model.mask_layer.weight.data

        def save_model(self, name):
            path = f"classifiers/saved_models/{name}"
            if isinstance(self.feature_extractor, nn.DataParallel):
                state = {
                    "feature_extractor": self.feature_extractor.module.state_dict(),
                    "classification_layer": self.classification_layer.module.state_dict(),
                    "mask_layer": self.mask_layer.module.state_dict()
                }
            else:
                state = {
                    "feature_extractor": self.feature_extractor.state_dict(),
                    "classification_layer": self.classification_layer.state_dict(),
                    "mask_layer": self.mask_layer.state_dict()
                }
            torch.save(state, path)

        def load_model(self, file_path, map_location=None):

            if map_location is None:
                state = torch.load(file_path, weights_only=True)
            else:
                state = torch.load(file_path, map_location=map_location, weights_only=True)

            if 'model' in state.keys(): # fix old state dicts so that they match new AbstractClassifier format
                classification_layer_state = {}
                classification_layer_state['weight'] = state['model']['fc.weight']
                classification_layer_state['bias'] = state['model']['fc.bias']
                del state['model']['fc.weight']
                del state['model']['fc.bias']
                state['feature_extractor'] = state.pop('model')
                state['classification_layer'] = classification_layer_state

            if state['classification_layer']['weight'].shape[0] != self.config.dataset.n_classes: # needed for backcompatibility with loaded Inception models
                state['classification_layer']['weight'] = state['classification_layer']['weight'][:self.config.dataset.n_classes]
                state['classification_layer']['bias'] = state['classification_layer']['bias'][:self.config.dataset.n_classes]

            if isinstance(self.feature_extractor, nn.DataParallel):
                self.feature_extractor.module.load_state_dict(state['feature_extractor'])
                self.classification_layer.module.load_state_dict(state['classification_layer'])
                self.mask_layer.module.load_state_dict(state['mask_layer'])
            else:
                self.feature_extractor.load_state_dict(state['feature_extractor'])
                self.classification_layer.load_state_dict(state['classification_layer'])
                self.mask_layer.load_state_dict(state['mask_layer'])


            if self.config.defense.onezero:
                self.mask_layer.weight.data = (~(torch.isclose(self.mask_layer.weight.data, torch.zeros_like(self.mask_layer.weight.data)))).float()

    drop_layer_defended_model = DropLayerClassifier(config)
    return drop_layer_defended_model


class ElementwiseLinear(nn.Module):
    def __init__(self, input_size: tuple, w_init=1) -> None:
        super(ElementwiseLinear, self).__init__()
        self.weight = nn.Parameter(torch.full(input_size, w_init), requires_grad=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # simple elementwise multiplication
        return self.weight * x


class MaskTransform(object):
    def __init__(self, mask_layer: nn.Module):
        self.mask_layer = mask_layer

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask_layer(x)