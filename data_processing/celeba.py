

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg
from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset, random_split
import torch

import numpy as np
import pandas as pd

from functools import partial
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
import os
from collections import Counter


import gdown
import zipfile
import requests
import shutil
from pathlib import Path

def download_from_google_drive(link, destination):
    """
    Download a file from Google Drive using the provided link.
    Handles files that require confirmation for large file downloads.
    
    :param link: Google Drive URL
    :param destination: Path to save the file
    """
    session = requests.Session()
    
    # Extract the file ID from the Google Drive link
    if "uc?id=" in link:
        file_id = link.split("uc?id=")[-1]
    elif "/d/" in link:
        file_id = link.split("/d/")[1].split("/")[0]
    else:
        raise ValueError("Invalid Google Drive link")
    
    URL = "https://drive.google.com/uc"
    response = session.get(URL, params={'id': file_id}, stream=True)
    response.raise_for_status()
    
    # Check for a confirmation token
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
        response.raise_for_status()
    
    # Download the file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download(dataset_folder="data", folder_name="celeba"):
    links = {
        # "img_align_celeba.zip": "https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ",
        "list_eval_partition.txt": "https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk/view?usp=sharing&resourcekey=0-i4TGCi_51OtQ5K9FSp4EDg",
        "list_landmarks_celeba.txt": "https://drive.google.com/file/d/0B7EVK8r0v71pTzJIdlJWdHczRlU/view?usp=sharing&resourcekey=0-49BtYuqFDomi-1v0vNVwrQ",
        "list_landmarks_align_celeba.txt": "https://drive.google.com/file/d/0B7EVK8r0v71pd0FJY3Blby1HUTQ/view?usp=sharing&resourcekey=0-aFtzLN5nfdhHXpAsgYA8_g",
        "list_bbox_celeba.txt": "https://drive.google.com/file/d/0B7EVK8r0v71pbThiMVRxWXZ4dU0/view?usp=sharing&resourcekey=0-z-17UMo1wt4moRL2lu9D8A",
        "list_attr_celeba.txt": "https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U/view?usp=sharing&resourcekey=0-YW2qIuRcWHy_1C2VaRGL3Q",
        "identity_CelebA.txt": "https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view?usp=sharing",
        "hdcrop.zip" : "https://drive.google.com/file/d/1f7RPM9iQL_3OCbcByDqjuZQ-9Qs_M_1A/view?usp=sharing"
    }
    
    target_folder = os.path.join(dataset_folder, folder_name)
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created directory: {target_folder}")
    else:
        print(f"Directory already exists: {target_folder}")
        
    for file_name, link in links.items():
        path_name = os.path.join(target_folder, file_name)
        
        if os.path.exists(path_name):
            print(f"{file_name} already exists in {target_folder}. Skipping download.")
            continue
        
        print(f"Downloading {file_name}...")
        
        file_extension = os.path.splitext(file_name)[1]
        
        if file_extension == ".zip":
            gdown.download(link, path_name, quiet = False, fuzzy=True)
            
        else:
            try:
                download_from_google_drive(link, path_name)
                print(f"Downloaded {file_name} to {path_name}")
            except Exception as e:
                print(f"Failed to download {file_name}. Error: {e}")
            
# def download(dataset_folder="data", folder_name = "celeba"):
#     links = {"img_align_celeba.zip" : "https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
#              ,"list_eval_partition.txt" : "https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk/view?usp=sharing&resourcekey=0-i4TGCi_51OtQ5K9FSp4EDg"
#              ,"list_landmarks_celeba.txt" : "https://drive.google.com/file/d/0B7EVK8r0v71pTzJIdlJWdHczRlU/view?usp=sharing&resourcekey=0-49BtYuqFDomi-1v0vNVwrQ"
#              ,"list_landmarks_align_celeba.txt" : "https://drive.google.com/file/d/0B7EVK8r0v71pd0FJY3Blby1HUTQ/view?usp=sharing&resourcekey=0-aFtzLN5nfdhHXpAsgYA8_g"
#              ,"list_bbox_celeba.txt" : "https://drive.google.com/file/d/0B7EVK8r0v71pbThiMVRxWXZ4dU0/view?usp=sharing&resourcekey=0-z-17UMo1wt4moRL2lu9D8A"
#              ,"list_attr_celeba.txt" : "https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U/view?usp=sharing&resourcekey=0-YW2qIuRcWHy_1C2VaRGL3Q"
#              ,"identity_CelebA.txt" : "https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view?usp=sharing"}
    
    
#     target_folder = os.path.join(dataset_folder, folder_name)
    
#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)
#         print(f"Created directory: {target_folder}")
#     else:
#         print(f"Directory already exists: {target_folder}")
        
    
#     for file_name, link in links.items():
        
#         path_name = os.path.join(target_folder, file_name)
        
#         if os.path.exists(path_name):
#             print(f"{file_name} already exists in {target_folder}. Skipping download.")
#             continue
        
#         gdown.download(link, path_name, quiet = False)
#         print(f"Downloaded {file_name} to {path_name}")


def process(dataset_folder="data", extract_folder = "celeba", file_name = "img_align_celeba.zip"):
    
    extract_folder = os.path.join(dataset_folder, extract_folder)
    unzipped_folder = os.path.join(extract_folder, os.path.splitext(file_name)[0])
    
    zip_file = os.path.join(extract_folder, file_name)
    
    if os.path.exists(unzipped_folder) and os.listdir(unzipped_folder):
        print(f"Extraction folder already contains data: {unzipped_folder}. Skipping unzipping.")
        return

    # Check if the ZIP file exists
    if os.path.isfile(zip_file):
        print(f"Extracting {zip_file} to {unzipped_folder}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(path = extract_folder)
            # with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            #     for member in zip_ref.infolist():
            #         # Skip directories
            #         if member.is_dir():
            #             continue
                    
            #         # Extract the file content
            #         file_data = zip_ref.read(member)

            #         # Get the file name only (drop the folder path)
            #         filename = Path(member.filename).name
                    
            #         # Write the file to the extract directory
            #         output_path = extract_folder / filename
            #         with open(output_path, 'wb') as f:
            #             f.write(file_data)
            print("Extraction complete.")
        except Exception as e:
            print(f"Error during extraction: {e}")
            
        # Move folder
        shutil.move(os.path.join(extract_folder, "data", "celeba", "hdcrop"), extract_folder)
        
        shutil.rmtree(os.path.join(extract_folder, "data"))
        
    else:
        print(f"ZIP file not found: {zip_file}")

class CelebA_N_most_common(Dataset):
    def __init__(self,
                 train,
                 split_seed=42,
                 transform=None,
                 root='data/celeba',
                 N = 1000,
                 img_folder: str = "img_align_celeba",
                 download: bool = False):
        # Load default CelebA dataset
        celeba = CustomCelebA(root=root,
                        split='all',
                        target_type="identity",
                        img_folder=img_folder)
        celeba.targets = celeba.identity

        # Select the N most frequent celebrities from the dataset
        targets = np.array([t.item() for t in celeba.identity])
        ordered_dict = dict(
            sorted(Counter(targets).items(),
                   key=lambda item: item[1],
                   reverse=True))
        sorted_targets = list(ordered_dict.keys())[:N]

        # Select the corresponding samples for train and test split
        indices = np.where(np.isin(targets, sorted_targets))[0]
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(indices))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        # Assert that there are no overlapping datasets
        assert len(set.intersection(set(train_idx), set(test_idx))) == 0

        # Set transformations
        self.transform = transform
        target_mapping = {
            sorted_targets[i]: i
            for i in range(len(sorted_targets))
        }

        self.target_transform = transforms.Lambda(lambda x: target_mapping[x])

        # Split dataset
        if train:
            self.dataset = Subset(celeba, train_idx)
            train_targets = np.array(targets)[train_idx]
            self.targets = [self.target_transform(t) for t in train_targets]
            self.name = 'CelebA1000_train'
        else:
            self.dataset = Subset(celeba, test_idx)
            test_targets = np.array(targets)[test_idx]
            self.targets = [self.target_transform(t) for t in test_targets]
            self.name = 'CelebA1000_test'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]


class CustomCelebA(VisionDataset):
    """ 
    Modified CelebA dataset to adapt for custom cropped images.
    """

    def __init__(
            self,
            root: str,
            split: str = "all",
            target_type: Union[List[str], str] = "identity",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            img_folder: str = "img_align_celeba",
    ):
        super(CustomCelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        
        self.img_folder = img_folder
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root)
        splits = pd.read_csv(fn("list_eval_partition.txt"), sep='\s+', header=None, index_col=0)
        identity = pd.read_csv(fn("identity_CelebA.txt"), sep='\s+', header=None, index_col=0)
        bbox = pd.read_csv(fn("list_bbox_celeba.txt"), sep='\s+', header=1, index_col=0)
        landmarks_align = pd.read_csv(fn("list_landmarks_align_celeba.txt"), sep='\s+', header=1)
        attr = pd.read_csv(fn("list_attr_celeba.txt"), sep='\s+', header=1)
        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor') # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_path = os.path.join(self.root, self.img_folder, self.filename[index])
        if os.path.exists(file_path) == False:
            file_path = file_path.replace('.jpg', '.png')
        X = PIL.Image.open(file_path)

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

if __name__ == "__main__":
    download(folder_name='celeba')
    
    process(extract_folder='celeba', file_name="hdcrop.zip")