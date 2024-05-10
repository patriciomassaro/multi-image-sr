from __future__ import annotations
from typing import List

import os
import glob

import rasterio
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset
import pandas as pd


from log.log_config import initialize_logger
from degradation_model.abstract_degradation_model import DegradationModel
from custom_dataclasses.hotspot_parameters import HotspotParameters,HotspotParametersRanges

class MISRDataset(Dataset):
    def __init__(self,
                 log_path: str,
                 folder_path: str,
                 degradation_model: DegradationModel,
                 images_to_keep:int,
                 hotspot_generation_parameter_ranges: HotspotParametersRanges
                 ):
        self.logger = initialize_logger(log_path)
        self.degradation_model = degradation_model
        self.hr_images_folder = folder_path
        self.images_to_keep = images_to_keep
        self.files = glob.glob(os.path.join(folder_path, "**", "*.tif"), recursive=True)
        self.hgpr = hotspot_generation_parameter_ranges
        if len(self.files) == 0:
            self.logger.error("No files found in " + self.hr_images_folder)
            raise ValueError("No files found in " + self.hr_images_folder)
        # Get the image size
        hr_rasterio_io = rasterio.open(self.files[0])
        self.image_size = hr_rasterio_io.read().squeeze().shape
        hr_rasterio_io.close()
        self.hgps = self._generate_random_hotspot_parameters()


    #define a function that will create a pandas dataframe for the dataset containing the information of the hotspot generation, according to the  HotspotParameters class
    def _generate_random_hotspot_parameters(self):
        n = self.__len__()
        p = self.hgpr.hotspot_probability
        max_hotspot_size_x = max(self.hgpr.hotspot_size_x_range)
        max_hotspot_size_y = max(self.hgpr.hotspot_size_y_range)

        return [
            HotspotParameters(
                add_hotspot_flag=random.choices([True, False], weights=[p,1-p])[0],
                max_intensity_multiplier=random.uniform(*self.hgpr.max_intensity_multiplier_range),
                hotspot_size_x=random.randint(*self.hgpr.hotspot_size_x_range),
                hotspot_size_y=random.randint(*self.hgpr.hotspot_size_y_range),
                sigma=random.uniform(*self.hgpr.sigma_range),
                hotspot_place_x= np.random.randint(2 * max_hotspot_size_x, self.image_size[0] - 2 * max_hotspot_size_x),
                hotspot_place_y= np.random.randint(2 * max_hotspot_size_y, self.image_size[1] - 2 * max_hotspot_size_y)
            )
            for i in range(n)
        ]
    
    def dump_hotspot_parameters(self):
        """ 
        Dump the hotspot parameters to a pandas dataframe
        """
        
        df = pd.DataFrame([vars(hgp) for hgp in self.hgps])
        return df
        

    def __len__(self):
        return len(self.files)

    @staticmethod
    def clip_image(image: np.ndarray, num_stds: int = 2.5) -> np.ndarray:
        """
        Clip image using mean and variance
        Args:
            image: image to clip
            num_stds: number of standard deviation to clip

        Returns:
            clipped image
        """
        image = np.clip(image, np.mean(image) - num_stds * np.std(image), np.mean(image) + num_stds * np.std(image))
        return image        

    def __getitem__(self, idx):
        # Load the image using rasterio
        hr_image_path = self.files[idx]

        # set the wavelenght based on if its lwir1 or lwir2
        if "lwir1" in hr_image_path.lower():
            wavelength = 8.7e-6
        elif "lwir2" in hr_image_path.lower():
            wavelength = 11.45e-6
        else:
            raise ValueError("wrong bands!!")

        hr_rasterio_io = rasterio.open(hr_image_path)
        hr_image = hr_rasterio_io.read().squeeze()
        hr_affine_transformation = hr_rasterio_io.transform
 
        hr_image,lr_images = self.degradation_model.process_image(hr_image=hr_image,
                                                                  affine_transform=hr_affine_transformation,
                                                                  wavelength=wavelength,
                                                                  hgp=self.hgps[idx])

        # take random T images
        random_idx = np.random.randint(0, lr_images.shape[0], self.images_to_keep)
        lr_images = lr_images[random_idx, :, :]

        # Normalize the images to 0 mean and 1 variance
        # Instance Normalization
        hr_image = (hr_image - np.nanmean(hr_image)) / np.nanstd(hr_image)
        lr_images = [(lr_image - np.nanmean(lr_image)) / np.nanstd(lr_image) for lr_image in lr_images]

        # Convert to torch
        hr_image = torch.from_numpy(hr_image)
        hr_image = hr_image.unsqueeze(0)

        lr_images = [torch.from_numpy(lr_image) for lr_image in lr_images]
        lr_images = torch.stack(lr_images)
        # LR images are a tensor of T X H X W, convert it to T X 1 X H X W
        lr_images = lr_images.unsqueeze(1)



        return {'hr_image':hr_image,
                'lr_images':lr_images,
                'file_path':hr_image_path}
    
    def plot_example(self, idx):
        data = self.__getitem__(idx)
        hr_image = data["hr_image"]
        lr_images = data["lr_images"]

        hr_image = hr_image.squeeze().cpu().numpy()
        hr_image = self.clip_image(hr_image)

        # normalize the image to [0,1]
        image = cv2.normalize(hr_image, None,
                              alpha=1e-3, beta=1 - 1e-3,
                              norm_type=cv2.NORM_MINMAX)

        # pick 3 images at random from the lr_images
        lr_images = lr_images.cpu().numpy()
        lr_images = lr_images[[0, 1, 2], :, :]
        lr_images = lr_images.squeeze()

        #  apply clip on every lr image while keeping them stacked in the same array
        lr_images = np.stack([self.clip_image(lr_image) for lr_image in lr_images])

        # normalize the lr
        lr_images = np.stack([cv2.normalize(lr_image, None,
                                            alpha=1e-3, beta=1 - 1e-3,
                                            norm_type=cv2.NORM_MINMAX) for lr_image in lr_images])

        # plot the images
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        cmap = 'inferno'
        # flatten the axs
        axs = axs.flatten()
        axs[0].imshow(hr_image, cmap=cmap)
        axs[0].set_title("HR image")
        axs[1].imshow(lr_images[0, :, :], cmap=cmap)
        axs[1].set_title("LR image 1")
        axs[2].imshow(lr_images[1, :, :], cmap=cmap)
        axs[2].set_title("LR image 2")
        axs[3].imshow(lr_images[2, :, :], cmap=cmap)
        axs[3].set_title("LR image 3")
        plt.show()

    def plot_lr_images_with_substracted_mean(self, idx):
        data = self.__getitem__(idx)
        print(data["file_path"])
        lr_images = data["lr_images"]
        
        # pick one of the lr_images at random and substract it to the rest
        random_idx = np.random.randint(0, lr_images.shape[0])
        random_pick = lr_images[random_idx, :, :, :]
        #Plot the random pick
        random_pick_numpy = random_pick.cpu().numpy()
        random_pick_numpy = random_pick_numpy.squeeze()
        random_pick_numpy = self.clip_image(random_pick_numpy)
        random_pick_numpy = cv2.normalize(random_pick_numpy, None,
                                          alpha=1e-3, beta=1 - 1e-3,
                                          norm_type=cv2.NORM_MINMAX)
        plt.imshow(random_pick_numpy, cmap='inferno')
        random_pick = random_pick.repeat(lr_images.shape[0],1,1,1)



        # plot the histogram of the intensities of the lr images before and after substracting the random pick
        fig, axs = plt.subplots(1, 2, figsize=(10,6))
        # flatten the axs
        axs = axs.flatten()
        axs[0].hist(lr_images.cpu().numpy().flatten(), bins=250)
        axs[0].set_title("histogram")

        lr_images = lr_images - random_pick 
        # drop the index that was substracted        
        lr_images = torch.cat([lr_images[:random_idx], lr_images[random_idx+1:]])
        axs[1].hist((lr_images).cpu().numpy().flatten(), bins=250)
        axs[1].set_title("histogram after substraction")
        plt.show()
        lr_images = lr_images.cpu().numpy()
        lr_images = lr_images.squeeze()
        lr_images = np.stack([cv2.normalize(lr_image, None,
                                        alpha=1e-3, beta=1 - 1e-3,
                                        norm_type=cv2.NORM_MINMAX) for lr_image in lr_images])

        # get number of images on the lr images
        n_images = lr_images.shape[0]

        #  apply clip on every lr image while keeping them stacked in the same array
        lr_images = np.stack([self.clip_image(lr_image) for lr_image in lr_images])

        # normalize the lr
        lr_images = np.stack([cv2.normalize(lr_image, None,
                                            alpha=1e-3, beta=1 - 1e-3,
                                            norm_type=cv2.NORM_MINMAX) for lr_image in lr_images])

        # Create subplots using the n_images divided in a squared grid
        fig, axs = plt.subplots(int(np.sqrt(n_images)), int(np.sqrt(n_images)), figsize=(20, 20))
        cmap = 'inferno'
        # flatten the axs
        axs = axs.flatten()
        # plot all the lr images
        for i, ax in enumerate(axs):
            ax.imshow(lr_images[i, :, :], cmap=cmap)
            ax.set_title("LR image {}".format(i + 1))
            

        plt.show()