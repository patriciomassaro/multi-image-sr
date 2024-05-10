import cv2
import torch
import datetime
from custom_dataclasses.hotspot_parameters import HotspotParametersRanges

# Training config
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_print_frequency= 150
val_print_frequency= train_print_frequency
seed=42
epochs = 60
train_batches_per_epoch = 1000000
validate_batches_per_epoch = 20000000

# RAMS parameters
T = 9
batch_size = 16
lr=7e-5
num_workers = 24
scale_factor=4
feature_attention_blocks=12

# Degradation model
source_resolution=70
target_resolution= source_resolution*scale_factor
source_error=0.5
target_error=1
interpolation_method=cv2.INTER_CUBIC
n_iterations=10
radiometric_error_correction_parameters = {"noise_factor": 0.5}
transfer_function_parameters = {"sigmaX_variance_param": 0.1,"sigmaY_variance_param": 0.1}
hotspot_generation_parameter_ranges= HotspotParametersRanges(
    hotspot_probability=0.05,
    max_intensity_multiplier_range=(1, 1.00001),
    hotspot_size_x_range=(1, 8),
    hotspot_size_y_range=(1, 8),
    sigma_range=(0.4, 0.45),
)

#Tensorboard 
tensorboard_experiment_name =  f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}-12blocks'
train_loss_name = "Train/Loss"
val_loss_name = "Val/Loss"
train_psnr_name = "Train/PSNR"
val_psnr_name = "Val/PSNR"
train_lpips_name= "Train/LPIPS"
val_lpips_name= "Val/LPIPS"
bicubic_psnr_name = "Val/Bicubic_PSNR"
bicubic_lpips_name = "Val/Bicubic_LPIPS"



#Checkpoints
best_file_name="model_best.pth.tar"
last_file_name="model_last.pth.tar"
samples_dir="samples"
results_dir="results"
checkpoints_dir="checkpoints"
pretrained_model_weights_path=""
# Write samples/ if you want to resume training
resume_training_weights_path=""
