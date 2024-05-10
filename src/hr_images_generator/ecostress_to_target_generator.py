from __future__ import annotations
from typing import List
import numpy as np
from typing import Union

from abstract_generator import AbstractGenerator


class EcostressTargetLWIRCreator(AbstractGenerator):
    """
    Leverages the AbstractGenerator class to create a generator for the ecostress data into a target mission.
    This mission has two bands:
    - LWIR1: can be calculated as the mean of the mapped radiance 1, 2 and 3
    - LWIR2: can be calculated as the mean of the mapped radiance 4 and 5

    The abstract methods implemented contain the logic to extract the metadata, verify the data is complete, and extract for this particular mission.
    """

    def _get_metadata_from_file(self, file_name) -> dict:
        year = file_name.split("/")[-2]
        location = file_name.split("/")[-3]
        image_type = file_name.split("/")[-4]
        metadata = {
            "year": year,
            "location": location,
            "image_type": image_type
        }
        return metadata

    def _verify_band_data_is_complete(self,
                                      group_dataset: dict,
                                      ) -> List[bool, bool]:
        lwir_1_verified = True
        lwir_2_verified = True
        bands_lwir_1 = ["mapped_radiance_1", "mapped_radiance_2", "mapped_radiance_3"]
        qa_bands_lwir_1 = ["mapped_data_quality_1", "mapped_data_quality_2", "mapped_data_quality_3"]
        bands_lwir_2 = ["mapped_radiance_4", "mapped_radiance_5"]
        qa_bands_lwir_2 = ["mapped_data_quality_4", "mapped_data_quality_5"]

        # Check if the strings in bands_lwir_1 and the qa_bands_lwir1 are in the group_dataset
        if not (
                all(band in group_dataset.keys() for band in bands_lwir_1) and
                all(band in group_dataset.keys() for band in qa_bands_lwir_1)
        ):
            lwir_1_verified = False
            self.logger.info(f"Missing LWIR1 bands in group")

        # Check if the strings in bands_lwir_2 and the qa_bands_lwir2 are in the group_dataset
        if not (
                all(band in group_dataset.keys() for band in bands_lwir_2) and
                all(band in group_dataset.keys() for band in qa_bands_lwir_2)
        ):
            lwir_2_verified = False
            self.logger.info(f"Missing LWIR2 bands in group")

        return lwir_1_verified, lwir_2_verified

    def _extract_stamp(self, file_name) -> List[str, str, str, str]:
        return file_name.split("_")[-2]

    def _extract_band(self, file_name) -> str:
        band = str(file_name.split("_", 1)[1])
        band = band.split("_doy")[0]
        return band.lower()

    def _verify_lwir_qa(self, group_dataset: dict) -> List[bool, bool]:
        lwir_1_verified = False
        lwir_2_verified = False
        bands_lwir_1 = [x + "_qa_passed" for x in self.lwir_1_bands]
        bands_lwir_2 = [x + "_qa_passed" for x in self.lwir_2_bands]

        # all the values of the keys must be true
        if all(group_dataset.get(band, False) for band in bands_lwir_1):
            lwir_1_verified = True
        if all(group_dataset.get(band, False) for band in bands_lwir_2):
            lwir_2_verified = True

        return lwir_1_verified, lwir_2_verified

    def _apply_QA_mask_to_image(self,
                                image: Union[np.ndarray, None],
                                qa: Union[np.ndarray, None],
                                ) -> List[np.ndarray, bool]:
        """
        Apply the QA mask to the raster, for the ecostress
        """
        if qa is None:
            return image, False, np.ones(image.shape) * -9997
        else:
            quality_passed_flag = True

            missing_data = (image == -9999.0) | (image == -9998.0)
            unseen = (image == -9997.0)
            bad_quality = (qa == 2) | (qa == 3) | (qa == 4)

            bad_pixels = missing_data | unseen | bad_quality

            total_bad_pixels = np.sum(bad_pixels)

            if total_bad_pixels / image.size > self.bad_pixels_threshold:
                quality_passed_flag = False
                return image, quality_passed_flag, bad_pixels

            image[bad_pixels] = np.nan

            image[bad_pixels] = np.nanmean(image)

            return image, quality_passed_flag, bad_pixels

    def _match_bands_with_qas(self, group_dataset) -> List[List]:
        """
        Match the bands with the quality bands
        :param group_dataset:
        :return:
        """
        quality_bands = [band for band in group_dataset.keys() if "quality" in band]

        radiance_bands = [band for band in group_dataset.keys() if "radiance" in band]

        radiances_dict = {rad.split('_')[-1]: rad for rad in radiance_bands}
        QA_dict = {qa.split('_')[-1]: qa for qa in quality_bands}
        combined = [[radiances_dict.get(id), QA_dict.get(id)] for id in set(radiances_dict) | set(QA_dict)]

        return combined
    