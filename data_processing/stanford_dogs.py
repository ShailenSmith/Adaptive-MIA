import os
import requests
import tarfile

from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import os
import numpy as np
import scipy.io
from PIL import Image

class StanfordDogs(Dataset):
    def __init__(self,
                 train,
                 cropped,
                 split_seed=42,
                 transform=None,
                 root='data/stanford_dogs'):

        self.image_path = os.path.join(root, 'Images')
        dataset = ImageFolder(root=self.image_path, transform=None)
        self.dataset = dataset
        self.cropped = cropped
        self.root = root

        self.annotations_folder = os.path.join(self.root, 'Annotation')
        self.breeds = os.listdir(self.image_path)

        self.classes = [cls.split('-', 1)[-1] for cls in self.dataset.classes]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        self.targets = self.dataset.targets
        self.name = 'stanford_dogs'

        split_train = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
        labels_train = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        split_test = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
        labels_test = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split_train] + [item[0][0] for item in split_test]
        labels = [item[0]-1 for item in labels_train] + [item[0]-1 for item in labels_test]

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx)
                                        for box in self.get_boxes(os.path.join(self.annotations_folder, annotation))]
                                        for annotation, idx in zip(split, labels)]
            self._flat_breed_annotations = [t[0] for t in self._breed_annotations]
            self.targets = [t[-1][-1] for t in self._breed_annotations]
            self._flat_breed_images = [(annotation+'.jpg', box, idx) for annotation, box, idx in self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation+'.jpg', idx) for annotation, idx in zip(split, labels)]
            self.targets = [t[-1] for t in self._breed_images]
            self._flat_breed_images = self._breed_images

        self.transform = transform
        indices = list(range(len(self.dataset)))
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(self.dataset))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        if train:
            self.dataset = np.array(self._flat_breed_images, dtype=object)[train_idx].tolist()
            self.targets = np.array(self.targets)[train_idx].tolist()
        else:
            self.dataset = np.array(self._flat_breed_images, dtype=object)[test_idx].tolist()
            self.targets = np.array(self.targets)[test_idx].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # im, _ = self.dataset[idx]
        image_name, target = self.dataset[idx][0], self.dataset[idx][-1]
        image_path = os.path.join(self.image_path, image_name)
        im = Image.open(image_path).convert('RGB')

        if self.cropped:
            im = im.crop(self.dataset[idx][1])
        if self.transform:
            return self.transform(im), target
        else:
            return im, target

    def get_boxes(self, path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes


def download(dataset_folder="data", folder_name = "stanford_dogs"):
    target_folder = os.path.join(dataset_folder, folder_name)
    
    # URLs for the files
    files = {
        "images.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
        "annotations.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar",
        "lists.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
        "train_data.mat": "http://vision.stanford.edu/aditya86/ImageNetDogs/train_data.mat",
        "test_data.mat": "http://vision.stanford.edu/aditya86/ImageNetDogs/test_data.mat",
    }

    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Download each file
    for file_name, url in files.items():
        file_path = os.path.join(target_folder, file_name)
        if os.path.exists(file_path):
            print(f"{file_name} already exists in {target_folder}. Skipping download.")
            continue
        try:
            print(f"Downloading {file_name} to {target_folder}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for HTTP codes 4xx/5xx
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
                    file.write(chunk)
            print(f"Downloaded {file_name} successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {file_name}: {e}")

def is_extraction_complete(target_folder):
    """
    Check if the extraction process is already complete.
    """
    expected_folders = ["Images", "Annotation"]
    expected_files = [
        "file_list.mat",
        "test_data.mat",
        "test_list.mat",
        "train_data.mat",
        "train_list.mat",
    ]
    
    # Check for expected folders
    for folder in expected_folders:
        folder_path = os.path.join(target_folder, folder)
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return False
    
    # Check for expected files
    for file_name in expected_files:
        file_path = os.path.join(target_folder, file_name)
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return False

    return True

def process(dataset_folder="data"):
    target_folder = os.path.join(dataset_folder, "stanford_dogs")
    
    # Check if already processed
    is_processed = is_extraction_complete(target_folder)
    
    if is_processed:
        print("Files already processed.")
        return 
    
    if not os.path.exists(target_folder):
        raise FileNotFoundError(f"The folder {target_folder} does not exist. Please run the download function first.")

    # Paths to tar files and the expected folder structure
    tar_files = [
        "images.tar",
        "annotations.tar",
        "lists.tar"
    ]
    
    # Extract tar files into the target folder
    for tar_file in tar_files:
        tar_path = os.path.join(target_folder, tar_file)
        extract_path = target_folder

        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"{tar_file} is missing in {target_folder}. Please check your download.")
        
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_path)
            print(f"{tar_file} extracted successfully.")

    # Ensure other files are in the right place
    expected_files = [
        "train_data.mat",
        "test_data.mat",
        "file_list.mat",
        "train_list.mat",
        "test_list.mat",
    ]
    
    for file_name in expected_files:
        file_path = os.path.join(target_folder, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_name} is missing in {target_folder}. Please check your download.")

    # # Cleanup: Remove tar files after extraction
    # for tar_file in tar_files.keys():
    #     tar_path = os.path.join(target_folder, tar_file)
    #     if os.path.exists(tar_path):
    #         os.remove(tar_path)
    #         print(f"Removed {tar_file} after extraction.")

    print("Folder structure has been successfully processed and verified.")





if __name__ == "__main__":
    download()
    process()