from __future__ import annotations
from typing import List, Any

import numpy as np
import cv2

from degradation_model.abstract_degradation_model import AbstractDegradationModel


class BasicDegradationModel(AbstractDegradationModel):

    def _pad_image(self,
                   image: np.ndarray,
                   padding: int) -> np.ndarray:
        return np.pad(array=image,
                      pad_width=padding,
                      mode='constant',
                      constant_values=np.mean(image)
                      )

    def _apply_transfer_function(self,
                                 image: np.ndarray,
                                 fx: float,
                                 fy: float,
                                 ) -> np.ndarray:
        sigma_x_variance_param = self.transfer_function_parameters.get("sigmaX_variance_param")
        sigma_y_variance_param = self.transfer_function_parameters.get("sigmaY_variance_param")
        sigma_x = (1 / fx - 1) / 2
        sigma_y = (1 / fy - 1) / 2

        sigmaX = sigma_x + np.random.normal(0, sigma_x_variance_param * sigma_x)
        sigmaY = sigma_y + np.random.normal(0, sigma_y_variance_param * sigma_y)

        image = np.squeeze(image)
        image = cv2.GaussianBlur(image, None, sigmaX=sigmaX, sigmaY=sigmaY,
                                 )

        return image

    def _quality_checks(self,
                        lr_images: List[np.ndarray],
                        ) -> bool:
        self._assert_nparray_not_equal(array_list=lr_images)

        return True

    def _apply_radiometric_error_correction(self,
                                            image: np.ndarray,
                                            wavelength: float,
                                            ) -> np.ndarray:
        noise_factor = self.radiometric_error_correction_parameters.get("noise_factor")

        radiometric_error = np.sqrt(np.abs(self.source_error ** 2 - self.target_error ** 2))
        # Convert to radiance using the derivative of the planck equation
        radiance_factor = (self._planck_equation(wav=wavelength, temperature=300 + 1e-6) - self._planck_equation(
            wav=wavelength, temperature=300)) / 1e-6
        radiance_error = radiometric_error * radiance_factor * 1e-6

        noise = np.random.normal(0, 1, image.shape)
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

        # Return the image with the noise
        return image + (1 - noise_factor) * radiance_error + noise_factor * noise * radiance_error