from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

import glob
import os

import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import math
import numpy as np
import concurrent.futures
import cv2


from log.log_config import initialize_logger


NORMALIZATION_MARGIN = 1e-3
DIVISION_EPSILON = 1e-7

@dataclass
class BandData:
    numpy_data: np.ndarray
    bounds: rasterio.coords.BoundingBox
    tf: rasterio.Affine
    crs: str
    image_size: int = None
    binary_mask: np.ndarray = None
    bad_pixel_count: int = None


class AbstractGenerator(ABC):
    """
    This abstract class represent the generation of from other missions

    The abstract methods must be defined in the children

    - _exctract_stamp: extracts the scene-stamp from the file name
    - _extract_band: extracts the band name from the full file name
    - _match_bands_with_qas: matches the radiance band with the QA band
    - _apply_QA_mask_to_raster: applies the QA mask to the raster
    - _convert_radiance_bands_to_target: converts the radiance bands to the two target bands
    """

    def __init__(self,
                 log_file_name:str,
                 input_folder: str,
                 output_folder: str,
                 lwir_1_bands: List[str],
                 lwir_2_bands: List[str],
                 bad_pixels_threshold: float,
                 crop_size: List[int,int],
                 num_crops: int,
                 ):

        self.logger = initialize_logger(log_file_name)

        self.lwir_1_bands = lwir_1_bands
        self.lwir_2_bands = lwir_2_bands
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.bad_pixels_threshold = bad_pixels_threshold
        
        self.logger.info("Input folder: " + self.input_folder)
        self.logger.info("Output folder: " + self.output_folder)
        self.files = glob.glob(os.path.join(self.input_folder,"**","*.tif"),recursive=True)
        if len(self.files) == 0:
            self.logger.error("No files found in " + self.input_folder)
            raise ValueError("No files found in " + self.input_folder)
        self.logger.info("Found " + str(len(self.files)) + " files in " + self.input_folder)

    @abstractmethod
    def _extract_stamp(self, file_name) -> str:
        """
        Mandatory function to be implemented by the child class, it extracts the scene-stamp from the file name
        :return: the stamp name
        """
        pass

    @abstractmethod
    def _extract_band(self, file_name) -> str:
        """
        Extract the band name from the full file name, must be implemented in the child
        :param file_name:
        :return: band name
        """
        pass

    @abstractmethod
    def _match_bands_with_qas(self, group_dataset) -> List[List]:
        """
        Matches the band with the QA band, must be implemented in the child class
        :param band:
        :return: a list of lists of size 2 with the band and the QA band names
        """
        pass

    @abstractmethod
    def _apply_QA_mask_to_image(self,
                                image: np.ndarray,
                                qa: np.ndarray) -> List[np.ndarray,bool,np.ndarray]:
        """
        Applies the QA mask to the raster, must be implemented in the child class
        :param raster:
        :param qa_band:
        :return:
        """
        pass

    @abstractmethod
    def _verify_band_data_is_complete(self,
                                      group_dataset: dict,
                                      ) -> List[bool,bool]:
        """
        Verifies that the band data is complete, must be implemented in the child class
        :param group_dataset:
        :return: True or False
        """
        pass

    @abstractmethod
    def _verify_lwir_qa(self,group_dataset:dict) -> List[bool,bool]:
        pass

    @staticmethod
    def save_raster(
            name_file: str,
            raster_to_save: np.ndarray,
            raster_transform: rasterio.Affine,
            target_crs: str) -> bool:
        """
        Saves the raster to the disk
        :param target_crs:
        :param filename:
        :param rasterio_object:
        :param raster_to_save:
        :param raster_transform:
        :return: True if the raster was saved successfully
        """
        with rasterio.open(
                fp=name_file,
                mode="w",
                driver="GTiff",
                width=raster_to_save.shape[1],
                height=raster_to_save.shape[0],
                count=1,
                dtype='float32',
                crs=target_crs,
                transform=raster_transform
        ) as dst:
            raster_to_save = np.expand_dims(raster_to_save, axis=0)
            dst.write(raster_to_save)

        return True

    @staticmethod
    def detect_stripe_noise(
        image: np.ndarray,
        min_num_modes: int = 1,
        max_num_modes: int = 2
    )-> bool:
        image = cv2.normalize(image,
                              None,
                              NORMALIZATION_MARGIN,
                              1-NORMALIZATION_MARGIN,
                              cv2.NORM_MINMAX)
        image = (image*255).astype(np.uint8)
        image = np.clip(image,0,255)

        thresh = cv2.adaptiveThreshold(image,
                                       255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,
                                       7,
                                       1)
        
        f = np.fft.fft2(thresh)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift)+1e-6)
        


        # Normalize the spectrum
        magnitude_spectrum = cv2.normalize(magnitude_spectrum,
                                           None,
                                           NORMALIZATION_MARGIN,
                                           1-NORMALIZATION_MARGIN,
                                           cv2.NORM_MINMAX)
        magnitude_spectrum = (magnitude_spectrum*255).astype(np.uint8)
        magnitude_spectrum = np.clip(magnitude_spectrum,0,255)
        magnitude_spectrum = 255-magnitude_spectrum
        # thresh spectrum
        _, magnitude_spectrum_thresh = cv2.threshold(magnitude_spectrum,
                                                     100,
                                                     255,
                                                     cv2.THRESH_BINARY)
        
           # detect points that form a line
        lines = cv2.HoughLinesP(255-magnitude_spectrum_thresh,
                                rho=5,
                                theta=np.pi/180,
                                threshold=30,
                                minLineLength=185,
                                maxLineGap=50)
        
            # detect unique angles
        angles = {}
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.round(np.arctan((y2-y1)/(x2-x1+DIVISION_EPSILON)), 2)
                if angles.get(f"{angle}"):
                    angles[f"{angle}"] += 1
                else:
                    angles[f"{angle}"] = 1

        if len(angles) >= min_num_modes and len(angles) <= max_num_modes:
            return True
        else:
            return False
    
    def _apply_quality_bands(self, group_dataset: dict) -> dict:
        """
        Applies the quality bands to the dataset
        :param group_dataset:
        :return:
        """
        for band, qa in self._match_bands_with_qas(group_dataset):
            band_data = group_dataset.get(band, None)
            qa_data = group_dataset.get(qa, None)
            if band_data is None or band_data is None:
                group_dataset[f"{band}_qa_passed"] = False
                continue
            band_data.numpy_data,quality_passed,band_data.binary_mask = self._apply_QA_mask_to_image(band_data.numpy_data,
                                                                                                    qa_data.numpy_data)
            if not quality_passed:
                group_dataset[f"{band}_qa_passed"] = False
            else:
                group_dataset[f"{band}_qa_passed"] = True
            del group_dataset[qa]

        lwir_1_qa_passed,lwir_2_qa_passed = self._verify_lwir_qa(group_dataset)

        return group_dataset,lwir_1_qa_passed,lwir_2_qa_passed

    def _get_random_crop(self,
                         band_data:BandData,
                         crop_size: [int, int],
                         seed:int,
                         max_retries:int=25):
        """
        Get a random crop from a bigger image. The crop should have ALL pixels with valid data.
        :param band_data:
        :param crop_size:
        :param seed:
        :return:
        """
        np.random.seed(seed)

        crop_h, crop_w = crop_size
        hr_image = band_data.numpy_data
        binary_mask = band_data.binary_mask
        original_tf = band_data.tf

        h, w = hr_image.shape
        valid_h = h - crop_h
        valid_w = w - crop_w

        if valid_h < 0 or valid_w < 0:
            raise ValueError(f"Invalid crop size {crop_size} for image with size {hr_image.shape}")

        tries = 0
        while True :
            top = np.random.randint(0, valid_h)
            left = np.random.randint(0, valid_w)

            binary_mask_region = binary_mask[top:top + crop_h, left:left + crop_w]
            if np.any(binary_mask_region==1):
                tries += 1
                if tries > max_retries:
                    return None,None
                # This means we have bad pixels in the region, we need to find another one
                continue

            stripe_noise_flag = self.detect_stripe_noise(hr_image[top:top + crop_h, left:left + crop_w])
            if stripe_noise_flag:
                self.logger.info("Stripe noise detected in the region,trying again")
                tries += 1
                if tries > max_retries:
                    self.logger.info("Max retries reached, returning None")
                    return None,None
                # This means we have stripe noise in the region, we need to find another one
                continue

            hr_image_region = hr_image[top:top + crop_h, left:left + crop_w]
            # calculate the new transform
            tf = original_tf * rasterio.Affine.translation(left, top)

            return hr_image_region,tf


    def _generate_random_crops(self,
                               band_data: BandData,
                               seed: int = 42):
        seeds = np.random.randint(0, 1000, self.num_crops)
        crops = []

        with concurrent.futures.ThreadPoolExecutor( ) as executor:
            results = [executor.submit(self._get_random_crop, band_data, self.crop_size, seed) for seed in seeds]
            for f in concurrent.futures.as_completed(results):
                crops.append(f.result())
        return crops



    def _get_all_stamps_metadata(self) -> dict:
        """
        Extracts all the stamps from the files without duplicates
        :return:
        """
        metadata_dict = []
        for file in self.files:
            metadata = self._get_metadata_from_file(file)
            metadata['stamp'] = self._extract_stamp(file)
            metadata_dict.append(metadata)
            #Remove duplicates
            metadata_dict = pd.DataFrame(metadata_dict).drop_duplicates().to_dict('records')
        # The stamps are in the format of YYYYMM
        return metadata_dict

    @abstractmethod
    def _get_metadata_from_file(self, file_name) -> dict:
        pass

    @staticmethod
    def _convert_radiance_bands_to_target_lwir(group_dataset: dict,bands_to_convert=List[str]) -> BandData:
        # Do a logical and of the binary masks in each BandData object
        binary_masks = [group_dataset[key].binary_mask for key in bands_to_convert]
        binary_mask = 1 * np.logical_and.reduce(binary_masks)

        data= np.mean(np.stack([group_dataset[key].numpy_data for key in bands_to_convert]),axis=0)

        data = BandData(
            numpy_data=data,
            image_size=data.size,
            tf=group_dataset[bands_to_convert[0]].tf,
            crs=group_dataset[bands_to_convert[0]].crs,
            bounds=group_dataset[bands_to_convert[0]].bounds,
            binary_mask=binary_mask,
            bad_pixel_count= binary_mask.size
        )
        return data

    def _get_group_from_stamp(self, stamp) ->List[str]:
        files = []
        for file in self.files:
            if self._extract_stamp(file) == stamp:
                files.append(file)
        return files

    def _load_group_dataset_dict(self, group):
        """
        Converts a group of file paths into a dictionary with the band as key and
        the rasterio file as value
        :param group:

        :return: group_dataset [dict]
        """
        files = self._get_group_from_stamp(group)
        if len(files) == 0:
            self.logger.error("No files found for stamp " + group)
            raise ValueError("No files found for stamp " + group)
        group_dataset = {}
        group_dataset["qa_passed_flag"] = True
        for file in files:
            band = self._extract_band(file)
            try:
                rasterio_file=rasterio.open(file)
                data = rasterio_file.read().squeeze().astype(np.float32)
            except:
                print("Error reading file " + file)
                self.logger.error("Error reading file " + file)
                # If we can't read the file, we delete it
                try:
                    os.remove(file)
                except:
                    self.logger.error("Error deleting file, it's already deleted " + file)
                # continue with the next file
                continue
            group_dataset[band] = BandData(
                numpy_data=data,
                image_size=data.size,
                bounds=rasterio_file.bounds,
                tf=rasterio_file.transform,
                crs=rasterio_file.crs
            )
        return group_dataset

    @staticmethod
    def _reproject_lwir_image(data:BandData) -> BandData:
        def _convert_wgs_to_utm(lon: float, lat: float):
            """Based on lat and lng, return best utm epsg-code"""
            utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
            if len(utm_band) == 1:
                utm_band = '0' + utm_band
            if lat >= 0:
                epsg_code = '326' + utm_band
                return epsg_code
            epsg_code = '327' + utm_band
            return epsg_code

        source_crs = "EPSG:" + str(data.crs.to_epsg())
        target_crs = "EPSG:" + _convert_wgs_to_utm(
            0.5 * (data.bounds.left + data.bounds.right),
            0.5 * (data.bounds.top + data.bounds.bottom)
        )
        # reproject the lwir image
        data.numpy_data, tf = reproject(
            data.numpy_data,
            src_crs=source_crs,
            src_transform=data.tf,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan
        )

        # reproject the qa binary mask
        data.binary_mask, data.tf = reproject(
            data.binary_mask.astype(np.float32),
            src_crs=source_crs,
            src_transform=data.tf,
            dst_crs=target_crs,
            dst_nodata=1.00,
            resampling=Resampling.nearest,
        )

        data.numpy_data = np.squeeze(data.numpy_data)
        data.binary_mask = np.squeeze(data.binary_mask)
        data.crs = target_crs

        return data



    def process_data(self):
        """
        Processes the data
        :return:
        """
        # Check if the output folder exists, otherwise create it
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        stamps_metadata = self._get_all_stamps_metadata()
        self.logger.info("Found " + str(len(stamps_metadata)) + " stamps")

        for stamps_metadata in stamps_metadata:

            LWIR_1 = None
            LWIR_2 = None


            # Load the data
            stamp = stamps_metadata['stamp']
            year = stamps_metadata['year']
            location = stamps_metadata['location']
            image_type = stamps_metadata['image_type']
            self.logger.info(f"Processing stamp {stamp} year {year} location {location} image_type {image_type}")

            group_dataset = self._load_group_dataset_dict(stamp)

            lwir_1_complete,lwir_2_complete = self._verify_band_data_is_complete(group_dataset)
            if not lwir_1_complete and not lwir_2_complete:
                self.logger.info("Skipping stamp " + stamp + " because the bands are not complete")
                continue

            group_dataset,lwir_1_qa_passed,lwir_2_qa_passed = self._apply_quality_bands(group_dataset)

            lwir_1_process = lwir_1_complete and lwir_1_qa_passed
            lwir_2_process = lwir_2_complete and lwir_2_qa_passed

            # skip if the quality bands are not complete or the qa was unsuccessful
            if not lwir_1_process and not lwir_2_process:
                self.logger.info("Skipping stamp " + stamp + " because the quality bands are not complete or the qa was unsuccessful")
                continue

            if lwir_1_process:
                self.logger.info("Processing LWIR1")
                LWIR_1 = self._convert_radiance_bands_to_target_lwir(group_dataset, bands_to_convert=self.lwir_1_bands)
                # Reproject only LWIR1 and it's QA
                LWIR_1 = self._reproject_lwir_image(LWIR_1)

                

            if lwir_2_process:
                self.logger.info("Processing LWIR2")
                LWIR_2 = self._convert_radiance_bands_to_target_lwir(group_dataset, bands_to_convert=self.lwir_2_bands)
                LWIR_2 = self._reproject_lwir_image(LWIR_2)
                

            for target_data, band_name in zip([LWIR_1,LWIR_2], ["LWIR1", "LWIR2"]):
                if target_data is None:
                    continue

                file_name = self.output_folder + "/" + image_type + "/" + location + "/" + year + "/" + stamp + "_" + band_name + ".tif"
                file_name_qa = self.output_folder + "/" + image_type + "/" + location + "/" + year + "/" + stamp + "_" + band_name +"_qa"+ ".tif"
                # check that the folder exists, otherwise create it
                if not os.path.exists(self.output_folder + "/" + image_type + "/" + location + "/" + year):
                    os.makedirs(self.output_folder + "/" + image_type + "/" + location + "/" + year)

                self.save_raster(
                    name_file=file_name,
                    raster_to_save=target_data.numpy_data,
                    raster_transform=target_data.tf,
                    target_crs=target_data.crs
                )
                
                self.save_raster(
                    name_file=file_name_qa,
                    raster_to_save=target_data.binary_mask,
                    raster_transform=target_data.tf,
                    target_crs=target_data.crs
                )

                # get random crops
                self.logger.info("Generating random crops")
                crops= self._generate_random_crops(band_data=target_data)

                if not os.path.exists(self.output_folder + "crops" + "/" + image_type + "/" + location + "/" + year):
                    os.makedirs(self.output_folder + "crops" + "/" + image_type + "/" + location + "/" + year)

                for i, crop_data in enumerate(crops):
                    crop, tf = crop_data
                    if crop is None:
                        continue
                    crop_file_name = self.output_folder + "crops" + "/" + image_type + "/" + location + "/" + year + "/" + stamp + "_" + band_name + "_" + str(i) + ".tif"

                    self.save_raster(
                        name_file=crop_file_name,
                        raster_to_save=crop,
                        target_crs=target_data.crs,
                        raster_transform=tf
                    )
        return 1