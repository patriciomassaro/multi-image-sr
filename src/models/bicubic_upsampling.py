import torch
import torch.nn as nn
import torch.nn.functional as F

class BicubicUpsampler(nn.Module):
    def __init__(self, scale_factor: int = 3):
        super(BicubicUpsampler, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, lr_batch: torch.tensor) -> torch.tensor:
        """
        Upsamples each image in the batch using bicubic interpolation.
        
        :param lr_batch: Batch of LR images of size N x 1 x H x W
        :return: Batch of upsampled images of size N x (scale_factor*H) x (scale_factor*W)
        """
        # Upsample using bicubic interpolation
        upsampled_batch = F.interpolate(lr_batch, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        upsampled_batch = upsampled_batch
        return upsampled_batch