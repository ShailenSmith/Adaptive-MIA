import gdown
import os
import shutil
from PIL import Image
from pathlib import Path
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

def download(dataset_folder="data", extract_folder="data/ffhq"):
    # List of individual Google Drive folder links or IDs (one per subfolder)
    subfolder_links = {
        # "00000": "https://drive.google.com/drive/folders/1taHKxS66YKJNhdhiGcEdM6nnE5W9zBb1",
        "01000": "https://drive.google.com/drive/folders/1QRt2us_EueWFilF-uJiwAbS4Jsb6IxAW",
        # "02000": "https://drive.google.com/drive/folders/1QXnHIoc_dmgzSdZQmQJ1I3QNOrRd7CSD",
        # "03000": "https://drive.google.com/drive/folders/1QHlEZpTtf-Bm9wH0pupFLwYU4R2fJi6e",
        # "04000": "https://drive.google.com/drive/folders/1tRvhNIMrkl6ypjKJN8Nh0Gr7gqZtPpc4",
        # "05000": "https://drive.google.com/drive/folders/1sHQK8mx8laO50_b828vQGtmfiQTqZ6Cm",
        # "06000": "https://drive.google.com/drive/folders/1s-c_sfM1Q1Or82gSW_INaT1v6dpAMY4J",
        # "07000": "https://drive.google.com/drive/folders/1sh5WdD5oR_1l1CcJ2OH2lqXzdH6G4T2K",
        # "08000": "https://drive.google.com/drive/folders/1rzpTdOU1xM_sySqHUMcnF6T5LLbaePxs",
        # "09000": "https://drive.google.com/drive/folders/1tOX9ovjkMmP0ayAtcLndJdKJtKbQ5CdO",
        # "10000": "https://drive.google.com/drive/folders/1sNpNx25nDQim0morw7LDPFIqXq5HYKyr",
        # "11000": "https://drive.google.com/drive/folders/1rxtwpfjNE3Ux7c_eC0ZT6G0ECLagGgFg",
        # "12000": "https://drive.google.com/drive/folders/1ta_GqlOa2TAzF1PCA-2RI4m9-Ux23MIT",
        # "13000": "https://drive.google.com/drive/folders/1t8Ii1qm63A15L0Wy7mrkYpEThMxD_abp",
        # "14000": "https://drive.google.com/drive/folders/1sM_TWzZ2MLMC3BicKWyeMQBVEwiCX41V",
        # "15000": "https://drive.google.com/drive/folders/1uGYCuEGuYpAIfB6O4_z27ZSmR5GfuyT1",
        # "16000": "https://drive.google.com/drive/folders/1v3QppxUVlMLuQeqDNXFotNJcboHaBvti",
        # add more subfolder links or IDs here
    }

    resized_dir = "data/ffhq_resized"
    temp_download = "data/temp_ffhq_download"
    Path(resized_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_download).mkdir(parents=True, exist_ok=True)

    # Authenticate PyDrive
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # This will open a browser to authenticate
    drive = GoogleDrive(gauth)

    def download_folder(folder_id, output_dir):
        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        for file in file_list:
            local_path = os.path.join(output_dir, file['title'])
            print(f"⬇️ Downloading {file['title']}")

            file.GetContentFile(local_path)

    for subfolder_link in subfolder_links.values():
        print(f"\n⏬ Downloading: {subfolder_link}")
        
        #gdown.download_folder(subfolder_link, output=temp_download, quiet=False, use_cookies=False)

        download_folder(subfolder_link.split('/')[-1], temp_download)

        print("🎨 Resizing images...")
        for root, dirs, files in os.walk(temp_download):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    orig_path = os.path.join(root, file)
                    rel_path = os.path.relpath(orig_path, temp_download)
                    new_path = os.path.join(resized_dir, rel_path)

                    os.makedirs(os.path.dirname(new_path), exist_ok=True)

                    try:
                        img = Image.open(orig_path).convert("RGB")
                        img = img.resize((224, 224), Image.LANCZOS)
                        img.save(new_path)
                    except Exception as e:
                        print(f"❌ Failed to process {orig_path}: {e}")

        print("🧹 Cleaning up...")
        #shutil.rmtree(temp_download)

    print("\n✅ All done!")
    
    
def resize_ffhq(dataset_folder="data/temp_ffhq_download/00000", extract_folder="data/ffhq/00000"):
    """
    Resizes images in the FFHQ dataset to 224x224 pixels.
    
    Args:
        dataset_folder (str): Path to the folder containing the FFHQ dataset.
        extract_folder (str): Path to the folder where resized images will be saved.
    """
    
    # Create the target directory if it doesn't exist
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
        print(f"Created directory: {extract_folder}")
    else:
        print(f"Directory already exists: {extract_folder}")
        
    # Resize images
    for root, dirs, files in tqdm(os.walk(dataset_folder), desc="Resizing images"):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                orig_path = os.path.join(root, file)
                rel_path = os.path.relpath(orig_path, dataset_folder)
                new_path = os.path.join(extract_folder, rel_path)

                os.makedirs(os.path.dirname(new_path), exist_ok=True)

                try:
                    img = Image.open(orig_path).convert("RGB")
                    img = img.resize((224, 224), Image.LANCZOS)
                    img.save(new_path)
                except Exception as e:
                    print(f"❌ Failed to process {orig_path}: {e}")

    print("✅ Resizing complete!")
    
class FFHQDataset(Dataset):
    def __init__(self, root, label=None, transform=None):
        self.root = root
        self.label = torch.tensor(label)
        self.transform = transform
        
        self.image_paths = list(Path(root).rglob("*.jpg")) + list(Path(root).rglob("*.png"))
        self.targets = torch.tensor([label] * len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, self.label
    
if __name__ == "__main__":
    # Example usage
    download()
    #resize_ffhq("data/temp_ffhq_download/00000", "data/ffhq_resized/00000")
    # dataset = FFHQDataset("data/ffhq_resized/00000", label=530)
    # print(f"Number of images: {len(dataset)}")
    # img, label = dataset[0]
    
    # print(f"Image size: {img.size}")
    # print(f"Label: {label}")
    