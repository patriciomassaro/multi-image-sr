import torch
import pytest
import random
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from models.loss import AdjustedPSNR,AdjustedL1Loss,AdjustedL2Loss,AdjustedLPIPS,_normalize_0_1

"""
Test Suite for Adjusted Image Comparison Metrics

This test suite aims to validate the functionality and effectiveness of custom image comparison metrics 
designed to handle slight spatial shifts in images. Specifically, the test cases cover:

1. AdjustedL1Loss: To verify if it's less than or equal to vanilla L1 Loss when spatial shifts exist.
2. AdjustedL2Loss: To verify if it's less than or equal to vanilla L2 (MSE) Loss under the same conditions.
3. AdjustedPSNR: To verify if it returns higher or equal PSNR compared to vanilla PSNR for spatially shifted images.
4. AdjustedLPIPS: To verify if it's less than or equal to vanilla LPIPS for spatially shifted images.

Each test uses randomized image shifts to ensure robustness.
"""


# Dummy data
y_true = torch.rand(5,1, 128, 128)
y_shifted = torch.rand(5,1, 128, 128).roll(shifts=(random.randint(3,4), random.randint(1,4)), dims=(2, 3))
# cap both images to [-1,1]
y_true,y_shifted= _normalize_0_1(y_true,y_shifted)


def test_adjusted_l1_loss():
    vanilla_l1 = torch.nn.L1Loss()(y_shifted, y_true)
    adjusted_l1 = AdjustedL1Loss()(y_shifted, y_true)
    assert adjusted_l1 <= vanilla_l1

def test_adjusted_l2_loss():
    vanilla_l2 = torch.nn.MSELoss()(y_shifted, y_true)
    adjusted_l2 = AdjustedL2Loss()(y_shifted, y_true)
    assert adjusted_l2 <= vanilla_l2

def test_adjusted_psnr():
    vanilla_psnr=PeakSignalNoiseRatio(data_range=1.0, base=10, dim=(1, 2, 3), reduction='none')(y_shifted, y_true)
    adjusted_psnr = AdjustedPSNR()(y_shifted, y_true)
    print(adjusted_psnr, vanilla_psnr)
    # check that every adjusted psnr is bigger than the vanilla psnr
    assert torch.all(adjusted_psnr >= vanilla_psnr) 

def test_adjusted_lpips():
    #repeat the tensors to 3 channels    
    vanilla_lpips = LearnedPerceptualImagePatchSimilarity(normalize=False)(y_shifted.repeat(1,3,1,1),
                                                                            y_true.repeat(1,3,1,1))
    adjusted_lpips = AdjustedLPIPS()(y_shifted, y_true)


    assert adjusted_lpips <= vanilla_lpips
