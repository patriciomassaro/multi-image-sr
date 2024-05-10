import os
import time
import random
from multiprocessing import cpu_count
from typing import List,Union,Tuple
num_workers = int(cpu_count()/4*3)
print("Number of workers used: ",num_workers)
import glob



import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models.RAMS import RAMS
from models.bicubic_upsampling import BicubicUpsampler
from log.log_config import initialize_logger
from log.tensorboard import TensorboardWriter
from degradation_model.basic_degradation_model import BasicDegradationModel
from models.loss import AdjustedPSNR,AdjustedL1Loss,AdjustedLPIPS,AdjustedL2Loss
import training_utils as training_utils
import training_config as training_config




def train(
        rams_model: RAMS,
        dataloader: DataLoader,
        device: torch.device,
        criterion: Union[nn.MSELoss,nn.L1Loss, AdjustedL1Loss, AdjustedL2Loss],
        psnr_model: nn.Module,
        lpips_model: nn.Module,
        optimizer: optim.Adam,
        epoch: int,
        writer: TensorboardWriter
) -> None:
    batches = len(dataloader)

    # Get progressbar
    batch_time = training_utils.AverageMeter("Time", ":6.3f")
    data_time = training_utils.AverageMeter("Data", ":6.3f")
    losses = training_utils.AverageMeter("Loss", ":6.6f")
    psnres = training_utils.AverageMeter("PSNR", ":4.2f")
    lpipses = training_utils.AverageMeter("LPIPS", ":4.2f")
    progress = training_utils.ProgressMeter(batches, [batch_time, data_time, losses,psnres,lpipses], prefix=f"Epoch: [{epoch + 1}]")

    # Initialize the data loader and load the first batch of data
    dataloader_iter = iter(dataloader)
    batch_index, batch_data = training_utils.initialize_batch(dataloader_iter=dataloader_iter)
    end = time.time()

    while batch_data is not None:
        if batch_index > training_config.train_batches_per_epoch:
            break

        # calculate the batch loading time
        data_time.update(time.time() - end)
        # get data from the batch
        hr_image = batch_data["hr_image"].to(device)
        lr_images = batch_data["lr_images"].to(device)

        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        sr = rams_model(lr_images)
        # Calculate the loss
        loss = criterion(sr, hr_image)
        # Backpropagate the loss
        loss.backward()
        # Update the weights
        optimizer.step()
        loss = loss.item()
        losses.update(loss, 1)
        if batch_index % training_config.train_print_frequency == 0:
            writer.log_scalar(training_config.train_loss_name, loss, epoch * batches + batch_index)

        with torch.no_grad():
            # calculate psnr and lpips
            psnr = psnr_model(sr, hr_image)
            lpips = lpips_model(sr, hr_image)
            # Get performance metric numbers
            psnr = psnr.mean().item()
            lpips = lpips.mean().item()


            # Update meters

            psnres.update(psnr, 1)
            lpipses.update(lpips, 1)
            # calculate the batch processing time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_index % training_config.train_print_frequency == 0:
                progress.display(batch_index + 1)
        # Preload the next batch of data
        batch_data, dataloader_iter = training_utils.preload_next_batch(dataloader_iter=dataloader_iter, dataloader=dataloader)
        batch_index += 1
    
    writer.log_scalar(training_config.train_psnr_name, psnr, epoch )
    writer.log_scalar(training_config.train_lpips_name, lpips, epoch)


def validate(
        rams_model: RAMS,
        dataloader: DataLoader,
        device: torch.device,
        criterion: Union[nn.MSELoss, nn.L1Loss],
        psnr_model: nn.Module,
        lpips_model: nn.Module,
        epoch: int,
        writer: TensorboardWriter,
        bicubic_model: BicubicUpsampler
) -> Tuple[float,float]:
    """
    Perform validation on the given data using the specified models and metrics.

    Args:
        rams_model (RAMS): The model used for super-resolution.
        dataloader (DataLoader): The data loader containing the validation data.
        device (torch.device): The device to perform the validation on.
        criterion (Union[nn.MSELoss, nn.L1Loss]): The loss criterion used for validation.
        psnr_model (nn.Module): The PSNR model used for evaluation.
        lpips_model (nn.Module): The LPIPS model used for evaluation.
        epoch (int): The current epoch number.
        writer (TensorboardWriter): The writer object for logging validation metrics.
        bicubic_model (BicubicUpsampler): The model used for bicubic upsampling.

    Returns:
        Tuple[float, float]: The average PSNR and LPIPS scores for the validation data.
    """
    
    batches = len(dataloader)
    # Calculate how many batches of data are in each Epoch
    batch_time = training_utils.AverageMeter("Time", ":6.3f")
    losses = training_utils.AverageMeter("Loss", ":4.4f")
    psnres = training_utils.AverageMeter("PSNR", ":4.2f")
    lpipses = training_utils.AverageMeter("LPIPS", ":4.2f")
    bicubic_psnres = training_utils.AverageMeter("Bicubic PSNR", ":4.2f")
    bicubic_lpipses = training_utils.AverageMeter("Bicubic LPIPS", ":4.2f")

    progress = training_utils.ProgressMeter(len(dataloader), [batch_time, losses,psnres,lpipses,bicubic_psnres,bicubic_lpipses], prefix="Test: ")

    # Initialize the data loader and load the first batch of data
    dataloader_iter = iter(dataloader)
    batch_index, batch_data = training_utils.initialize_batch(dataloader_iter=dataloader_iter)
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            if batch_index > training_config.validate_batches_per_epoch:
                break
            hr_image = batch_data["hr_image"].to(device)
            lr_images = batch_data["lr_images"].to(device)

            # pick a random candidate out of the T images in the batch of N X T X H X W
            random_index = random.randint(0, training_config.T-1) 
            lr_image = lr_images[:,random_index,:,:]
            lr_image_upsampled = bicubic_model(lr_image)
            bicubic_psnr = psnr_model(lr_image_upsampled, hr_image)
            bicubic_lpips = lpips_model(lr_image_upsampled, hr_image)

            # model
            sr = rams_model(lr_images)

            psnr = psnr_model(sr, hr_image)
            lpips = lpips_model(sr, hr_image)
            loss = criterion(sr, hr_image)

            #other metrics comes out as a tensor, we reduce it
            loss = loss.item()
            psnr = psnr.mean().item()
            lpips = lpips.mean().item()
            bicubic_lpips = bicubic_lpips.mean().item()
            bicubic_psnr = bicubic_psnr.mean().item()

            losses.update(loss, 1)
            psnres.update(psnr,1)
            lpipses.update(lpips,1)
            bicubic_psnres.update(bicubic_psnr,1)
            bicubic_lpipses.update(bicubic_lpips,1)
            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_index % training_config.val_print_frequency == 0:
                progress.display(batch_index + 1)
            # Preload the next batch of data
            batch_data, dataloader_iter = training_utils.preload_next_batch(dataloader_iter=dataloader_iter,
                                                                   dataloader=dataloader)

            # HR and LR remain untouched
            if epoch == 0:
                writer.log_images(f"Val/Image/{batch_index}_HR",
                                   training_utils.convert_torch_tensor_to_image(hr_image[0]),
                                   epoch)
                # writer.log_images(f"Val/Image/{batch_index}_LRS",
                #                    training_utils.convert_torch_tensor_to_image(lr_images[0]),
                #                    training_step)
                writer.log_images(f"Val/Image/{batch_index}_LR",
                                   training_utils.convert_torch_tensor_to_image(lr_image_upsampled[0]),
                                   epoch)
            # SR changes over epochs
            writer.log_images(f"Val/Image/{batch_index}_SR", training_utils.convert_torch_tensor_to_image(sr[0]), epoch)

            batch_index += 1

    progress.display_summary()
    writer.log_scalar(training_config.val_loss_name,losses.avg, epoch )
    writer.log_scalar(training_config.val_psnr_name, psnres.avg, epoch)
    writer.log_scalar(training_config.val_lpips_name, lpipses.avg, epoch)
    writer.log_scalar(training_config.bicubic_psnr_name, bicubic_psnres.avg, epoch)
    writer.log_scalar(training_config.bicubic_lpips_name, bicubic_lpipses.avg, epoch)


    return psnres.avg, lpipses.avg


def main():
    print("Initializing logs and tensorboard, check logs folder for more info")
    # get parent directory of this file
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Initiallize the logger
    logger = initialize_logger(os.path.join(parent_dir, "logs", "train.log"))
    # Initialize the tensorboard writer as None
    tensorboard_writer = None


    # Initialize the number of training epochs and loss
    start_epoch = 0
    best_psnr = 0.0

    logger.info("Initializing training")

    device = training_config.device
    # log the device
    logger.info("Using device: " + str(device))
    # Initialize the random seed
    torch.manual_seed(training_config.seed)
    logger.info("Using random seed: " + str(training_config.seed))

    basic_degradation_model = BasicDegradationModel(
        log_file_name=os.path.join(parent_dir,'logs',"basic_degradation_model.log"),
        source_resolution=training_config.source_resolution,
        target_resolution=training_config.target_resolution,
        source_error=training_config.source_error,
        target_error=training_config.target_error,
        interpolation_method=training_config.interpolation_method,
        n_iterations=training_config.n_iterations,
        radiometric_error_correction_parameters=training_config.radiometric_error_correction_parameters,
        transfer_function_parameters=training_config.transfer_function_parameters
    )

    # Log the degradation model and its parameters
    logger.info("Degradation model Initialized")

    train_dataloader, val_dataloader = training_utils.load_dataset(degradation_model=basic_degradation_model,
                                                          batch_size=training_config.batch_size,
                                                          num_workers= num_workers,
                                                          T=training_config.T)
    logger.info("dataloaders loaded")

    # Log the dataset length
    logger.info(" Training Dataset length: " + str(len(train_dataloader.dataset)))
    logger.info(" Validation Dataset length: " + str(len(val_dataloader.dataset)))

    # Initialize the network
    model = RAMS(scale_factor=training_config.scale_factor,
                 t=training_config.T,
                 c=1,
                 num_feature_attn_blocks=training_config.feature_attention_blocks)
    model.to(device)
    bicubic_model = BicubicUpsampler(scale_factor=training_config.scale_factor)
    bicubic_model.to(device)

    logger.info("Model initialized")
    logger.info("Model parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    criterion = AdjustedL2Loss().to(device=device)
    optimizer = optim.Adam(model.parameters(),lr=training_config.lr)
    adjusted_psnr_model = AdjustedPSNR().to(device=device).eval()
    adjusted_lpips_model = AdjustedLPIPS().to(device=device).eval()

    logger.info("Loss and optimizer initialized")

    print("Check whether to load pretrained model weights...")
    if training_config.pretrained_model_weights_path:
        # look for the last checkpoint file in the samples folder
        if os.path.exists(os.path.join(parent_dir, training_config.samples_dir)):
            #load all files with blob and get the last one
            checkpoint_file = max(glob.glob(os.path.join(parent_dir, training_config.samples_dir, "*.pth.tar")),
                                  key=os.path.getctime)
            model = training_utils.load_checkpoint(model, checkpoint_file)
            print(f"Loaded `{checkpoint_file}` pretrained model weights successfully.")

    
    print("Check whether to resume training...")
    if training_config.resume_training_weights_path:
        # look for the last checkpoint file in the samples folder
        if os.path.exists(os.path.join(parent_dir, training_config.samples_dir)):
            checkpoints = glob.glob(os.path.join(parent_dir, training_config.samples_dir, "*.pth.tar"))
            if len(checkpoints) != 0:
                #load all files with blob and get the last one
                checkpoint_file = max(checkpoints,key=os.path.getctime)
                model,_,start_epoch,best_psnr,lpips,optimizer,_,tensorboard_writer = training_utils.load_checkpoint(model,
                                                                                                checkpoint_file,
                                                                                                optimizer=optimizer,
                                                                                                load_mode='resume'
                                                                                                )
                print(f"Loaded `{checkpoint_file}` pretrained model weights successfully.")

    
    if not tensorboard_writer:
        tensorboard_writer = TensorboardWriter(os.path.join(parent_dir, "logs", "tensorboard",training_config.tensorboard_experiment_name))

    
    tensorboard_writer.log_pandas_dataframe(tag="train_hotspot_config",
                                            dataframe=train_dataloader.dataset.dump_hotspot_parameters(),
                                            step=0)
    tensorboard_writer.log_pandas_dataframe(tag="val_hotspot_config",
                                            dataframe=val_dataloader.dataset.dump_hotspot_parameters(),
                                            step=0)

    for epoch in range(start_epoch, training_config.epochs):
        print("Epoch: " + str(epoch))
        
        train(rams_model=model,
                        dataloader=train_dataloader,
                        device=device,
                        criterion=criterion,
                        optimizer=optimizer,
                        psnr_model=adjusted_psnr_model,
                        lpips_model=adjusted_lpips_model,
                        epoch=epoch,
                        writer=tensorboard_writer,
                    )
        psnr,lpips = validate(rams_model=model,
                        dataloader=val_dataloader,
                        device=device,
                        criterion=criterion,
                        psnr_model=adjusted_psnr_model,
                        lpips_model=adjusted_lpips_model,
                        epoch=epoch,
                        writer=tensorboard_writer,
                        bicubic_model=bicubic_model
                        )
        
        file_path = os.path.join(parent_dir, "nohup.out")
        with open(file_path, "r") as file:
            file_contents = file.read()
        tensorboard_writer.log_text("nohup.out", file_contents,1)

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        is_last = (epoch + 1) == training_config.epochs
        best_psnr = max(psnr, best_psnr)
        training_utils.save_checkpoint(state_dict={
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_psnr": best_psnr,
            "lpips": lpips,
            "optimizer": optimizer.state_dict(),
            "log_dir" : tensorboard_writer.log_dir
        },
            is_best=is_best,
            is_last=is_last,
            file_name=f"checkpoint_{epoch}.pth.tar",
            results_dir=os.path.join(parent_dir, training_config.results_dir),
            samples_dir=os.path.join(parent_dir, training_config.samples_dir),
            best_file_name=training_config.best_file_name,
            last_file_name=training_config.last_file_name
        )


if __name__ == "__main__":
    main()
