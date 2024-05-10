from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any

import rasterio
import cv2
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

from log.log_config import initialize_logger
from custom_dataclasses.hotspot_parameters import HotspotParameters


class AbstractDegradationModel(ABC):

    def __init__(self,
                 log_file_name: str,
                 source_resolution: int,
                 target_resolution: int,
                 source_error: float,
                 target_error: float,
                 interpolation_method: int,
                 transfer_function_parameters: dict[str, Any],
                 radiometric_error_correction_parameters: dict[str, Any],
                 n_iterations: int, 
                 ):
        self.logger = initialize_logger(log_file_name)

        if source_resolution > target_resolution:
            self.logger.error("Source resolution must be greater than target resolution")
            raise Exception("Source resolution must be greater than target resolution")
        self.target_resolution = target_resolution
        self.source_resolution = source_resolution
        self.resolution_ratio = target_resolution / source_resolution
        self.source_error = source_error
        self.target_error = target_error
        self.n_iterations = n_iterations
        self.transfer_function_parameters = transfer_function_parameters
        self.radiometric_error_correction_parameters = radiometric_error_correction_parameters
        self.interpolation_method = interpolation_method

    @abstractmethod
    def _apply_radiometric_error_correction(self,
                                            image: np.ndarray,
                                            wavelength: float,
                                            ) -> np.ndarray:
        """
        Apply the radiometric error correction, that is defined in the children
        :param image: the raster in numpy
        :param wavelength:  the average wavelength of the band
        :return: the degraded image
        """
        pass

    @abstractmethod
    def _pad_image(self,
                   image: np.ndarray,
                   padding:int) -> np.ndarray:
        """
        Pads the image, the type of padding is defined in the children classes
        :param image:
        :param dx:
        :param dy:
        :return:
        """
        pass
        # # Add a padding to the image, with the value of the neighborhood mean

    @abstractmethod
    def _apply_transfer_function(self,
                                 image: np.ndarray,
                                 fx: float,
                                 fy: float
                                 ) -> np.ndarray:
        """
        Applies the transfer function from our system to the dataset, implemented in children
        """
        pass

    @abstractmethod
    def _quality_checks(self,
                        lr_images: List[np.ndarray],
                        ) -> bool:
        pass

    @staticmethod
    def _plot_band(data: np.ndarray,
                   title: str,
                   ax: plt.Axes,
                   clip: tuple = (30, 70)):
        data = np.clip(data, np.percentile(data, clip[0]), np.percentile(data, clip[1]))
        ax.imshow(data, cmap='inferno')
        ax.set_title(title)

    @staticmethod
    def _assert_nparray_not_equal(array_list: List[np.ndarray]) -> bool:
        for i in range(len(array_list)):
            for j in range(i + 1, len(array_list)):
                if np.array_equal(array_list[i], array_list[j]):
                    raise Exception(f"LR images {i} and {j} are equal")
        return True

    @staticmethod
    def _planck_equation(wav: float,
                         temperature: float) -> float:
        """
        Planck's equation to calculate the intensity of a black body
        :param wav:
        :param temperature:
        :return intensity:
        """
        a = 2.0 * sc.Planck * sc.speed_of_light ** 2
        b = sc.Planck * sc.speed_of_light / (wav * sc.Boltzmann * temperature)
        intensity = a / ((wav ** 5) * (np.exp(b) - 1.0))
        return intensity

    @staticmethod
    def _resize_image(image: np.ndarray,
                      fx: float,
                      fy: float,
                      original_affine: rasterio.Affine,
                      interpolation_method: int) -> [np.ndarray, rasterio.Affine]:

        """
        Resizes the image and returns the affine transformation matrix
        """
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation_method)

        # Define the new transform
        new_transform = rasterio.Affine(
            original_affine[0] / fx,
            original_affine[1],
            original_affine[2],
            original_affine[3],
            original_affine[4] / fy,
            original_affine[5],
        )
        return image, new_transform
    
    @staticmethod
    def _add_hotspot(image:np.ndarray,
                    hgp: HotspotParameters,
                     )-> np.ndarray:
        """
        Add a hotspot to the image, must be implemented in the child
        :param image: the image to add the hotspot to
        :return: the image with the hotspot
        """
        def _calculate_mean_of_borders(y_start, x_start, y_size, x_size):
            # Calculate the mean of the borders
            mean_top = np.nanmean(image[y_start:y_start + 1, x_start:x_start + x_size])
            mean_bottom = np.nanmean(image[y_start + y_size - 1:y_start + y_size, x_start:x_start + x_size])
            mean_left = np.nanmean(image[y_start:y_start + y_size, x_start:x_start + 1])
            mean_right = np.nanmean(image[y_start:y_start + y_size, x_start + x_size - 1:x_start + x_size])
            return np.mean([mean_top, mean_bottom, mean_left, mean_right])
        
        # Create a gaussian hotspot of size 2*hotspot_size + 1
        y, x = np.meshgrid(
            np.linspace(-1, 1, 2 * hgp.hotspot_size_x + 1),
            np.linspace(-1, 1, 2 * hgp.hotspot_size_y + 1)
        )
        d = np.sqrt(x * x + y * y)
        mu = 0
        hotspot = np.exp(-((d - mu) ** 2 / (2.0 * hgp.sigma ** 2)))
        hotspot = hotspot / np.max(hotspot)
        
        max_intensity = np.nanmax(image)
        min_intensity = _calculate_mean_of_borders(y_start=hgp.hotspot_place_y,
                                                    x_start=hgp.hotspot_place_x,
                                                    x_size=2 * hgp.hotspot_size_x + 1,
                                                    y_size=2 * hgp.hotspot_size_y + 1)

        # Normalize the hotspot like a 0-1 but using max intensity and minimum intensity of the image
        hotspot = hotspot * (max_intensity * hgp.max_intensity_multiplier - min_intensity) + min_intensity

        # Add the hotspot to the image
        image[
            hgp.hotspot_place_y - hgp.hotspot_size_y:hgp.hotspot_place_y + hgp.hotspot_size_y + 1,
            hgp.hotspot_place_x - hgp.hotspot_size_x:hgp.hotspot_place_x + hgp.hotspot_size_x + 1
             ] = hotspot

        return image

    def _apply_subpixel_shift(self,
                              image: np.ndarray,
                              dx: int,
                              dy: int,
                              padding:int) -> np.ndarray:
        """
        Applies a subpixel shift to the image
        :param image: image to shift
        :param dx: shift in x
        :param dy: shift in y
        """
        # Make sure that the shift is not bigger than a pixel in the image when we downsample
        if abs(dx) > self.resolution_ratio or abs(dy) > self.resolution_ratio:
            self.logger.error("Shift is bigger than a pixel in the image when we downsample")
            raise Exception("Shift is bigger than a pixel in the image when we downsample")

        m = np.float32([[1, 0, dx], [0, 1, dy]])
        image = cv2.warpAffine(src=image, M=m,
                            dsize=(image.shape[1], image.shape[0]),
                               borderMode=cv2.BORDER_REFLECT_101
                               )

        return image

    def process_image(self,
                       hr_image: np.ndarray,
                       affine_transform: rasterio.Affine,
                       wavelength: float,
                       hgp: HotspotParameters,
                       ) -> List[np.ndarray]:
        """
        Resamples the dataset to the target resolution, applying the degradation model
        :param hr_image: the high resolution image
        :param affine_transform:
        :param wavelength:
        :return: the low resolution image and the list of low resolution images
        """
        if hgp.add_hotspot_flag:
            hr_image = self._add_hotspot(image=hr_image,
                                        hgp=hgp)

        
        lr_images = []

        
        fx = fy= self.source_resolution / self.target_resolution
        random_shifts = np.random.randint(low=int(-self.resolution_ratio),
                                          high=int(self.resolution_ratio),
                                          size=(self.n_iterations * 20, 2))
        # Drop duplicates
        random_shifts = np.unique(random_shifts, axis=0)
        # Delete shift that is (0,0)
        random_shifts = random_shifts[np.logical_not(np.all(random_shifts == 0, axis=1))]

        np.random.shuffle(random_shifts)
        if random_shifts.shape[0] < self.n_iterations:
            self.logger.warning(
                f"Not possible to generate {self.n_iterations} random shifts, using only {random_shifts.shape[0]}")
        else:
            random_shifts = random_shifts[:self.n_iterations]

        # get the max shift possible
        max_shift = np.max(np.abs(random_shifts))

        for dx,dy in random_shifts:

            lr_image = self._apply_subpixel_shift(image=hr_image,
                                                  dx=dx,
                                                  dy=dy,
                                                  padding=max_shift,
                                                  )
            lr_image = self._apply_transfer_function(image=lr_image,
                                                     fy=fy,
                                                     fx=fx,
                                                     )
            lr_image, _ = self._resize_image(image=lr_image,
                                             fx=fx,
                                             fy=fy,
                                             interpolation_method=self.interpolation_method,
                                             original_affine=affine_transform)
            lr_image = self._apply_radiometric_error_correction(image=lr_image,
                                                                wavelength=wavelength
                                                                )
            lr_image = lr_image.astype(np.float32)

            lr_images.append(lr_image)

        self._quality_checks(lr_images=lr_images)
        lr_images = np.stack(lr_images)
        return hr_image,lr_images
