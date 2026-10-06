
#! WIP -- don't push to public yet
'''
    Source: https://github.com/mseitzer/pytorch-fid
    Modified code to be compatible with our attack pipeline

    Copyright [2021] [Maximilian Seitzer]

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

'''

import numpy as np
import pytorch_fid.fid_score
import torch
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm

from Plug_and_Play_Attacks.datasets.custom_subset import SingleClassSubset

IMAGE_EXTENSIONS = ('bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp')


class FID_Score:
    def __init__(self, dataset_1, dataset_2, device, batch_size=128, dims=2048, num_workers=8, gpu_devices=[]):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.batch_size = batch_size
        self.dims = dims
        self.num_workers = num_workers
        self.device = device

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        inception_model = InceptionV3([block_idx]).to(self.device)
        if len(gpu_devices) > 1:
            self.inception_model = torch.nn.DataParallel(inception_model, device_ids=gpu_devices)
        else:
            self.inception_model = inception_model
        self.inception_model.to(device)

    def compute_fid(self, rtpt=None):
        m1, s1 = self.compute_statistics(self.dataset_1, rtpt)
        m2, s2 = self.compute_statistics(self.dataset_2, rtpt)
        fid_value = pytorch_fid.fid_score.calculate_frechet_distance(
            m1, s1, m2, s2)
        return fid_value

    def compute_fid_by_class(self):
        """Compute per-class FID between dataset_1 (real) and dataset_2 (fake).

        Requires dataset_1.targets and dataset_2.targets to use the same class IDs.
        Classes with fewer than 2 samples in either dataset are skipped.
        Note: per-class FID is unreliable with very few samples.
        """
        targets = np.array(self.dataset_2.targets)
        target_classes = np.unique(targets).tolist()
        fid_by_class = {}
        for cls in tqdm(target_classes, desc='FID by class'):
            real_subset = SingleClassSubset(self.dataset_1, target_class=int(cls))
            fake_subset = SingleClassSubset(self.dataset_2, target_class=int(cls))
            if len(real_subset) < 2 or len(fake_subset) < 2:
                continue
            m1, s1 = self.compute_statistics(real_subset)
            m2, s2 = self.compute_statistics(fake_subset)
            fid_by_class[int(cls)] = pytorch_fid.fid_score.calculate_frechet_distance(
                m1, s1, m2, s2)
        return fid_by_class

    def compute_statistics(self, dataset, rtpt=None):
        self.inception_model.eval()
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers)
        pred_arr = np.empty((len(dataset), self.dims))
        start_idx = 0
        max_iter = int(len(dataset) / self.batch_size)
        for step, (x, y) in enumerate(dataloader):
            with torch.no_grad():
                x = x.to(self.device)
                pred = self.inception_model(x)[0]
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

            if rtpt:
                rtpt.step(
                    subtitle=f'FID Score Computation step {step} of {max_iter}')

        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
        return mu, sigma
