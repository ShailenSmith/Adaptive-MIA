import numpy as np
import pytorch_fid.fid_score
import torch
from pytorch_fid.inception import InceptionV3
from Plug_and_Play_Attacks.utils.stylegan import create_image
from Plug_and_Play_Attacks.metrics.fid_score import FID_Score
from Plug_and_Play_Attacks.datasets.custom_subset import SingleClassSubset

IMAGE_EXTENSIONS = ('bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp')


class FID_Score_by_target(FID_Score):
    def __init__(self, dataset_1, dataset_2, device, crop_size=None, generator=None, batch_size=128, dims=2048, num_workers=8, gpu_devices=[]):
        super().__init__(dataset_1, dataset_2, device, crop_size=crop_size, generator=generator, 
                         batch_size=batch_size, dims=dims, num_workers=num_workers, gpu_devices=gpu_devices)
    
    def compute_fid_by_class(self):
        final_targets = self.dataset_2.targets
        target_classes=torch.unique(final_targets).cpu().tolist()
        
        training_subset_by_target = [SingleClassSubset(self.dataset_1, target_class=target_class) for target_class in target_classes]
        attack_subset_by_target = [SingleClassSubset(self.dataset_2, target_class=target_class) for target_class in target_classes]
        
        fid_values = []
        for target_dataset, attack_dataset in zip(training_subset_by_target, attack_subset_by_target):
            m1, s1 = self.compute_statistics(target_dataset)
            m2, s2 = self.compute_statistics(attack_dataset)
            fid_value = pytorch_fid.fid_score.calculate_frechet_distance(
                m1, s1, m2, s2)
            
            fid_values.append(fid_value)
            
        return fid_values, target_classes
            
        

    