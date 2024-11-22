import os
import glob
import numpy as np
import torch
import pandas as pd
import natsort
import pickle
from os.path import join
import cv2

def pickle_dump(file, path):
    """ Function to dump pickle object """
    with open(path, 'wb') as f:
        pickle.dump(file, f, -1)

def clamp(image):
    image = np.minimum(255, np.maximum(0, image))
    assert image.max() > 0
    return np.uint8(image)

class GenerateCrops:
    def __init__(self, crop_size, device):
        self.crop_size = crop_size
        self.device = device

    def split_image(self, image):
        crop_size = self.crop_size
        burst_size, channels, imsize1, imsize2 = image.shape
        npatch1 = imsize1 // crop_size
        npatch2 = imsize2 // crop_size
        image = image[:, :, :npatch1 * crop_size, :npatch2 * crop_size]
        image = image.reshape(burst_size, channels, npatch1, crop_size, npatch2, crop_size)
        image = image.permute(0, 2, 4, 1, 3, 5)
        image = image.reshape(burst_size, npatch1 * npatch2, channels, crop_size, crop_size)
        image = image.permute(1, 0, 2, 3, 4)
        return image

    def __call__(self, burst_path):
        print(f"Processing: {burst_path} on {self.device}")
        imgs_path = glob.glob(join(burst_path, 'aligned', '*.jpg'))
        imgs_path = natsort.natsorted(imgs_path)

        # Load target image
        target = cv2.imread(join(burst_path, 'helicon_focus.jpg'), cv2.IMREAD_COLOR)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        dim = (int(target.shape[1]), int(target.shape[0]))

        images = []
        for i, path in enumerate(imgs_path):
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            images.append(image)
        images = np.array(images)

        assert all([images[0].shape == x.shape for x in images])
        assert target.shape == images[0].shape

        images = torch.Tensor(images).to(torch.uint8).permute(0, 3, 1, 2).to(self.device)
        target = torch.Tensor(target)[None].to(torch.uint8).permute(0, 3, 1, 2).to(self.device)

        images = self.split_image(images)
        target = self.split_image(target)

        assert images.shape[0] == target.shape[0], f'{images.shape}'
        assert images.shape[0] > 0

        outpath = join(burst_path, 'crops3')
        os.makedirs(outpath, exist_ok=True)
        for i, (burst_crop, target_crop) in enumerate(zip(images, target)):
            burst_crop = burst_crop.clone()
            target_crop = target_crop.clone()
            data = {'burst': burst_crop, 'target': target_crop}
            torch.save(data, join(outpath, f'crop{i}.pkl'))

if __name__ == '__main__':
    crop_size = 128
    df = pd.read_csv('dataset.csv', sep=";")

    for split in ['train', 'test']:
        # Filter the dataset
        split_df = df[df['set'] == split][['lens', 'photo']]
        bursts_list = split_df.apply(lambda x: join(split, x[0], x[1]), axis=1).values
        bursts_list = list(bursts_list)

        # Assign GPUs
        gpu_devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        gpu_jobs = {0: [], 1: []}

        # Distribute jobs to GPUs
        for idx, burst_path in enumerate(bursts_list):
            gpu_jobs[idx % 2].append(burst_path)

        # Process jobs on each GPU
        for gpu_idx, gpu_burst_paths in gpu_jobs.items():
            device = gpu_devices[gpu_idx]
            generator = GenerateCrops(crop_size, device)
            for burst_path in gpu_burst_paths:
                generator(burst_path)
