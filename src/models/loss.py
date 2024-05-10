from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn as nn
import torch

def _crop_image_to_pixel_shift(y_pred:torch.Tensor,
                               y:torch.Tensor,
                               max_pixel_shift:int,
                               i:int,
                               j:int) -> [torch.Tensor, torch.Tensor]:
    image_size = y_pred.shape[2]
    cropped_y_pred = y_pred[:, :,
                     i:i + (image_size - max_pixel_shift),
                     j:j + (image_size - max_pixel_shift)]
    cropped_y = y[:, :,
                i:i + (image_size - max_pixel_shift),
                j:j + (image_size - max_pixel_shift)]
    return cropped_y_pred, cropped_y

def _correct_bias(y_pred:torch.Tensor,
                    y:torch.Tensor) -> torch.Tensor:

    bias = (y - y_pred).mean(dim=[2, 3], keepdim=True)
    bias = bias.expand(-1, -1, y_pred.shape[2], y.shape[3])
    y_pred = y_pred + bias

    return y_pred

def _normalize_0_1(y_pred:torch.Tensor,
                     y:torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    # get min and max for each element in batch
    vmin = torch.min(y.reshape(y.size(0), -1), dim=1).values
    vmin = vmin.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(y_pred)
    vmax = torch.max(y.reshape(y.size(0), -1), dim=1).values
    vmax = vmax.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(y_pred)

    # scale to common 0 - 1 scale to compare
    cropped_y_pred_scaled = (y_pred - vmin) / (vmax - vmin)
    cropped_y_scaled = (y - vmin) / (vmax - vmin)

    return cropped_y_pred_scaled, cropped_y_scaled




class AdjustedPSNR(nn.Module):
    def __init__(self, max_pixel_shift=6):
        super(AdjustedPSNR, self).__init__()
        self.model = PeakSignalNoiseRatio(data_range=1.0,
                                          base=10,
                                          dim=(1, 2, 3),
                                          reduction='none')
        self.max_pixel_shift = max_pixel_shift

    def forward(self, y_pred, y):
        if y_pred.shape != y.shape:
            raise ValueError("Shapes must be equal!")
        if len(y_pred.shape) != 4:
            raise ValueError("Input shape must have 4 dimensions (batch, channel, height, width).")

        candidates = []
        for i in range(self.max_pixel_shift + 1):
            for j in range(self.max_pixel_shift + 1):
                # Crop the images
                cropped_y_pred, cropped_y = _crop_image_to_pixel_shift(y_pred, y, self.max_pixel_shift, i, j)
                # Apply bias correction
                cropped_y_pred = _correct_bias(cropped_y_pred, cropped_y)
                # Normalize to 0-1
                cropped_y_pred_scaled, cropped_y_scaled = _normalize_0_1(cropped_y_pred, cropped_y)

                candidates.append(self.model(cropped_y_pred_scaled,cropped_y_scaled))

        # Stack the candidates along a new dimension
        candidates_tensor = torch.stack(candidates, dim=-1)
        # Compute the maximum PSNR for each image across all N candidates
        max_psnrs = candidates_tensor.max(dim=-1).values
        return max_psnrs

class AdjustedL1Loss(nn.Module):
    def __init__(self, max_pixel_shift=6):
        super(AdjustedL1Loss, self).__init__()
        self.model = nn.L1Loss()
        self.max_pixel_shift = max_pixel_shift

    def forward(self, y_pred, y):
        if y_pred.shape != y.shape:
            raise ValueError("Shapes must be equal!")
        if len(y_pred.shape) != 4:
            raise ValueError("Input shape must have 4 dimensions (batch, channel, height, width).")

        candidates = []
        for i in range(self.max_pixel_shift + 1):
            for j in range(self.max_pixel_shift + 1):
                # Crop the images
                cropped_y_pred, cropped_y = _crop_image_to_pixel_shift(y_pred, y, self.max_pixel_shift, i, j)
                # Apply bias correction
                cropped_y_pred = _correct_bias(cropped_y_pred, cropped_y)
                candidates.append(self.model(cropped_y_pred, cropped_y))

        # Pick the result with the lower L1 Loss

        # Stack the candidates along a new dimension
        candidates_tensor = torch.stack(candidates, dim=-1)
        # Compute the minimum L1 for each image across all N candidates
        min_l1 = candidates_tensor.min(dim=-1).values 
        return min_l1
    
class AdjustedL2Loss(nn.Module):
    def __init__(self, max_pixel_shift=6):
        super(AdjustedL2Loss, self).__init__()
        self.model = nn.MSELoss()
        self.max_pixel_shift = max_pixel_shift

    def forward(self, y_pred, y):
        if y_pred.shape != y.shape:
            raise ValueError("Shapes must be equal!")
        if len(y_pred.shape) != 4:
            raise ValueError("Input shape must have 4 dimensions (batch, channel, height, width).")

        candidates = []
        for i in range(self.max_pixel_shift + 1):
            for j in range(self.max_pixel_shift + 1):
                # Crop the images
                cropped_y_pred, cropped_y = _crop_image_to_pixel_shift(y_pred, y, self.max_pixel_shift, i, j)
                # Apply bias correction
                cropped_y_pred = _correct_bias(cropped_y_pred, cropped_y)
                candidates.append(self.model(cropped_y_pred, cropped_y))

        # Pick the result with the lower L1 Loss

        # Stack the candidates along a new dimension
        candidates_tensor = torch.stack(candidates, dim=-1)
        # Compute the minimum L1 for each image across all N candidates
        min_l2 = candidates_tensor.min(dim=-1).values # Do not use .values here, we need the Loss object
        return min_l2

class AdjustedLPIPS(nn.Module):
    def __init__(self, max_pixel_shift=6):    
        super(AdjustedLPIPS, self).__init__()
        self.model = LearnedPerceptualImagePatchSimilarity(normalize=False)
        self.max_pixel_shift = max_pixel_shift

    def forward(self, y_pred, y):
        if y_pred.shape != y.shape:
            raise ValueError("Shapes must be equal!")
        if len(y_pred.shape) != 4:
            raise ValueError("Input shape must have 4 dimensions (batch, channel, height, width).")

        candidates = []
        for i in range(self.max_pixel_shift + 1):
            for j in range(self.max_pixel_shift + 1):
                # Crop the images
                cropped_y_pred, cropped_y = _crop_image_to_pixel_shift(y_pred, y, self.max_pixel_shift, i, j)
                # Apply bias correction
                cropped_y_pred = _correct_bias(cropped_y_pred, cropped_y)
                #Normalize
                cropped_y_pred_scaled, cropped_y_scaled = _normalize_0_1(cropped_y_pred, cropped_y)
                # clipped the values to -1, 1
                cropped_y_pred_scaled = torch.clamp(cropped_y_pred_scaled, -1, 1)
                # Replicate the channels to match the LPIPS input
                cropped_y_pred_scaled = cropped_y_pred_scaled.repeat(1, 3, 1, 1)
                cropped_y_scaled = cropped_y_scaled.repeat(1, 3, 1, 1)

                candidates.append(self.model(cropped_y_pred_scaled, cropped_y_scaled))
        # Stack the candidates along a new dimension
        candidates_tensor = torch.stack(candidates, dim=-1)
        # Compute the minimum lpips for each image across all N candidates
        min_lpips = candidates_tensor.min(dim=-1).values # Do not use .values here, we need the Loss object
        return min_lpips



