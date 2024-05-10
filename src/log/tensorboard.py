import os
from tensorboardX  import SummaryWriter
import training.training_config as tc


METRICS_LAYOUT = {
    "Plots":{
        "Loss" : ["Multiline", [tc.train_loss_name, tc.val_loss_name]],
        "PSNR" : ["Multiline",[tc.train_psnr_name, tc.val_psnr_name, tc.bicubic_psnr_name]],
        "LPIPS" : ["Multiline",[tc.train_lpips_name, tc.val_lpips_name, tc.bicubic_lpips_name]],
        }
    }


class TensorboardWriter:
    """
    Class to abstract all tensorboard operations
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        # check if log_dir exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.writer.add_custom_scalars(METRICS_LAYOUT)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag, values, step, bins='tensorflow'):
        self.writer.add_histogram(tag, values, step, bins=bins)

    def log_images(self, tag, images, step):
        # This assumes that images is a mini-batch of images
        self.writer.add_images(tag, images, step)
    
    def log_image(self,tag,image,step):
        self.writer.add_image(tag,image,step)

    def log_model(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)

    def log_text(self, tag, text, step):
        self.writer.add_text(tag, text, step)

    def log_figure(self, tag, figure, step):
        self.writer.add_figure(tag, figure, step)

    def log_text_at_global_step(self, tag, text, step):
        self.writer.add_text(tag, text, global_step=step)

    def log_hparams(self, hparam_dict, metric_dict):
        self.writer.add_hparams(hparam_dict, metric_dict)

    def log_pandas_dataframe(self, tag, dataframe, step):
        
        self.writer.add_text(tag, dataframe.to_html(), step)

    def close(self):
        self.writer.close()