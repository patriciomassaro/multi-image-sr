from __future__ import annotations
from enum import Enum
import cv2
import torch
from torch.utils.data import DataLoader
from datasets.misr_dataset import MISRDataset
import os
import shutil
from log.tensorboard import TensorboardWriter
import training.training_config as tc

def choice_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available()  else "cpu")

def load_dataset(degradation_model,
                 T:int,
                 num_workers:int,
                 batch_size=16) -> DataLoader:
    # Get the parent directory of this file
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_dataset= MISRDataset(log_path=os.path.join(parent_dir,"src","training","logs","train_dataset.log"),
                               folder_path=os.path.join(parent_dir, "data","training", "DATASET_NAME"),
                               degradation_model=degradation_model,
                               images_to_keep = T,
                               hotspot_generation_parameter_ranges=tc.hotspot_generation_parameter_ranges)
    
    val_dataset = MISRDataset(log_path=os.path.join(parent_dir,"src","training","logs","val_dataset.log"),
                              folder_path=os.path.join(parent_dir, "data", "validation", "DATASET_NAME"),
                              degradation_model=degradation_model,
                              images_to_keep=T,
                              hotspot_generation_parameter_ranges=tc.hotspot_generation_parameter_ranges)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  drop_last=True,
                                  persistent_workers=False)
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  drop_last=True,
                                  persistent_workers=False)
    return train_dataloader,val_dataloader

def clip_to_sigma(
        tensor:torch.Tensor,
        sigma:int =2.0
)-> torch.Tensor:
    """
    Clips the values of the tensor to sigma standard deviations
    :param tensor: The tensor to clip
    :param sigma: The number of standard deviations to clip to
    :return: The clipped tensor
    """
    mean = torch.nanmean(tensor)
    std = torch.nanstd(tensor)
    return torch.clamp(tensor, mean - sigma * std, mean + sigma * std)

def convert_torch_tensor_to_image(
        tensor:torch.Tensor,
        normalize:bool=True
) -> torch.Tensor:

    # If the tensor is 3D, then it is an HR image 1 X H X W,
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    if normalize:
        tensor = ( tensor-tensor.min()) / (tensor.max()-tensor.min())
    return tensor


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def preload_next_batch(dataloader_iter: torch.DataLoaderIter,
                       dataloader: torch.DataLoader):
    try:
        batch_data = next(dataloader_iter)
    except StopIteration:
        batch_data = None
    except RuntimeError:
        print(f"Undefined error in batch-loading, restarting dataloader.")
        dataloader_iter = iter(dataloader)
        batch_data = next(dataloader_iter)
    return batch_data,dataloader_iter

def initialize_batch(dataloader_iter:torch.DataLoaderIter):
    try:
    # Initialize the number of data batches to print logs on the terminal
        batch_index = 0
        batch_data = next(dataloader_iter)
    except RuntimeError as e:
        batch_index = 1
        batch_data = next(dataloader_iter)
    return batch_index, batch_data

def calculate_progress(
        epoch,
        batches
) -> ProgressMeter:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    return ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_checkpoint(
        model: torch.nn.Module,
        weights_path: str,
        ema_model: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        load_mode="pretrain"):
    checkpoint = torch.load(weights_path, map_location= lambda storage, loc: storage)
    
    if load_mode=="resume":
        start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        lpips = checkpoint["lpips"]
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        if scheduler is not None:
            # Load the scheduler model
            scheduler.load_state_dict(checkpoint["scheduler"])

        if ema_model is not None:
            # Load ema model state dict. Extract the fitted model weights
            ema_model_state_dict = ema_model.state_dict()
            ema_state_dict = {k: v for k, v in checkpoint["ema_state_dict"].items() if k in ema_model_state_dict.keys()}
            # Overwrite the model weights to the current model (ema model)
            ema_model_state_dict.update(ema_state_dict)
            ema_model.load_state_dict(ema_model_state_dict)
        # create the summary writer from the log dir
        tensorboard_writer = TensorboardWriter(checkpoint["tensorboard_dir"])

        return model,ema_model,start_epoch,best_psnr,lpips,optimizer,scheduler,
    else:
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
        # Overwrite the model weights to the current model
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
        return model




def save_checkpoint(
        state_dict: dict,
        file_name: str,
        samples_dir: str,
        results_dir: str,
        best_file_name: str,
        last_file_name: str,
        is_best: bool = False,
        is_last: bool = False,
) -> None:

    checkpoint_path = os.path.join(samples_dir, file_name)
    # check if the directory exists
    make_directory(samples_dir)
    make_directory(results_dir)
    torch.save(state_dict, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, best_file_name))
    if is_last:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, last_file_name))