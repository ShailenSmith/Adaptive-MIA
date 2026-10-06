import os
import zipfile
import subprocess

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import ImageFolder

def download(dataset_folder="data", filename="facescrub-full.zip"):
    # Full path to the dataset file
    file_path = os.path.join(dataset_folder, filename)

    # Check if the file already exists
    if os.path.isfile(file_path):
        print(f"Dataset already exists at: {file_path}")
    else:
        print(f"Downloading dataset to: {file_path}")
        # Use curl to download the dataset
        url = "https://www.kaggle.com/api/v1/datasets/download/rajnishe/facescrub-full"
        try:
            subprocess.run(
                ["curl", "-L", "-o", file_path, url],
                check=True
            )
            print("Download complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error during download: {e}")

def process_zip(dataset_folder="data", extract_folder = "data/facescrub", filename="facescrub-full.zip"):
    
    """
    Creates a directory and extracts the ZIP file into it.

    Args:
        zip_file (str): Path to the ZIP file.
        extract_folder (str): Destination folder to extract files.
    """
    
    zip_file = os.path.join(dataset_folder, filename)
    
    # Check if extraction folder already has content
    if os.path.exists(extract_folder) and os.listdir(extract_folder):
        print(f"Extraction folder already contains data: {extract_folder}. Skipping unzipping.")
        return
    
    # Create the target directory if it doesn't exist
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
        print(f"Created directory: {extract_folder}")
    else:
        print(f"Directory already exists: {extract_folder}")
    
    # Check if the ZIP file exists
    if os.path.isfile(zip_file):
        print(f"Extracting {zip_file} to {extract_folder}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            print("Extraction complete.")
        except Exception as e:
            print(f"Error during extraction: {e}")
    else:
        print(f"ZIP file not found: {zip_file}")


class FaceScrub(Dataset):
    def __init__(self,
                 group,
                 train,
                 split_seed=42,
                 transform=None,
                 root='data/facescrub'):

        if group == 'actors':

            root = os.path.join(root, 'actor_faces')
            
            self.dataset = ImageFolder(root=root, transform=None)
            self.targets = self.dataset.targets
            self.name = 'facescrub_actors'

        elif group == 'actresses':
    
            root = os.path.join(root, 'actress_faces')
    
            self.dataset = ImageFolder(root=root, transform=None)
            self.targets = self.dataset.targets
            self.name = 'facescrub_actresses'

        elif group == 'all':

            root_actors = os.path.join(root, 'actor_faces')
            root_actresses = os.path.join(root, 'actress_faces')
   
            dataset_actors = ImageFolder(root=root_actors, transform=None)
            target_transform_actresses = lambda x: x + len(dataset_actors.
                                                           classes)
            dataset_actresses = ImageFolder(
                root=root_actresses,
                transform=None,
                target_transform=target_transform_actresses)
            dataset_actresses.class_to_idx = {
                key: value + len(dataset_actors.classes)
                for key, value in dataset_actresses.class_to_idx.items()
            }
            self.dataset = ConcatDataset([dataset_actors, dataset_actresses])
            self.classes = dataset_actors.classes + dataset_actresses.classes
            self.class_to_idx = {
                **dataset_actors.class_to_idx,
                **dataset_actresses.class_to_idx
            }
            self.targets = dataset_actors.targets + [
                t + len(dataset_actors.classes)
                for t in dataset_actresses.targets
            ]
            self.name = 'facescrub_all'

        else:
            raise ValueError(
                f'Dataset group {group} not found. Valid arguments are \'all\', \'actors\' and \'actresses\'.'
            )

        self.transform = transform
        indices = list(range(len(self.dataset)))
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(self.dataset))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        if train:
            self.dataset = Subset(self.dataset, train_idx)
            self.targets = np.array(self.targets)[train_idx].tolist()
        else:
            self.dataset = Subset(self.dataset, test_idx)
            self.targets = np.array(self.targets)[test_idx].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]

def main():
    download()
    process_zip()


if __name__ == "__main__":
    main()