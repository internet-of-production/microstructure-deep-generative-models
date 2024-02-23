"""File where parameters for (VAE) experiments are defined."""

import os


class Hyperparams(dict):
    """Class to store hyperparameters.

    From: https://github.com/openai/vdvae/blob/main/hps.py
    """
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


def add_parser_arguments(parser):
    parser.add_argument("--path_data", type=str)
    parser.add_argument("--path_figures", type=str)
    parser.add_argument("--path_model", type=str)
    parser.add_argument("--path_results", type=str)
    parser.add_argument("--path_analysis_file", type=str)
    parser.add_argument("--path_log_dir", type=str)
    parser.add_argument("--device", default="cpu", type=str)


# 100 Microns
vae_hparams_100x100 = Hyperparams()
vae_hparams_100x100.z_dim = 32
vae_hparams_100x100.h_dim = 8_192
vae_hparams_100x100.num_classes = 9
vae_hparams_100x100.num_epochs = 500
vae_hparams_100x100.beta = 1
vae_hparams_100x100.learn_rate = 1e-4
vae_hparams_100x100.batch_size = 32
vae_hparams_100x100.shuffle_data = True
vae_hparams_100x100.input_size = (32, 1, 512, 512)
vae_hparams_100x100.enc = (
    # in_channels, out_channels, kernel_size, stride, padding
    ["Conv2d", (3, 32, 3, 2, 1)],
    # ["BatchNorm2d", (32,)],
    ["ReLU", ()],

    ["Conv2d", (32, 32, 3, 2, 1)],
    # ["BatchNorm2d", (64,)],
    ["ReLU", ()],

    # ["MaxPool2d", (2, 2)],

    ["Conv2d", (32, 32, 3, 2, 1)],
    # ["BatchNorm2d", (128,)],
    ["ReLU", ()],

    # ["MaxPool2d", (2, 2)],

    ["Conv2d", (32, 32, 3, 2, 1)],
    # ["BatchNorm2d", (256,)],
    ["ReLU", ()],

    ["Conv2d", (32, 32, 3, 2, 1)],
    # ["BatchNorm2d", (512,)],
    ["ReLU", ()],

)
vae_hparams_100x100.dec = (
    ["Unflatten", (32, 16, 16)],
    # in_channels, out_channels, kernel_size, stride, padding, output_padding
    ["ConvTranspose2d", (32, 32, 4, 2, 1, 0)],
    # ["BatchNorm2d", (256,)],
    ["ReLU", ()],

    # ["Upsample", (None, 2)],

    ["ConvTranspose2d", (32, 32, 4, 2, 1, 0)],
    # ["BatchNorm2d", (128,)],
    ["ReLU", ()],

    # ["Upsample", (None, 2)],

    ["ConvTranspose2d", (32, 32, 3, 2, 1, 1)],
    # ["BatchNorm2d", (64,)],
    ["ReLU", ()],

    ["ConvTranspose2d", (32, 32, 4, 2, 1, 0)],
    # ["BatchNorm2d", (32,)],
    ["ReLU", ()],

    ["ConvTranspose2d", (32, 1, 4, 2, 1, 0)],
    ["Sigmoid", ()],
)

# 200 Microns
vae_hparams_200x200 = Hyperparams()
vae_hparams_200x200.h_dim = 32_768  # 4096
vae_hparams_200x200.z_dim = 32
vae_hparams_200x200.num_classes = 9
vae_hparams_200x200.num_epochs = 500
vae_hparams_200x200.beta = 1
vae_hparams_200x200.learn_rate = 1e-4
vae_hparams_200x200.batch_size = 32
vae_hparams_200x200.shuffle_data = True
vae_hparams_200x200.input_size = (32, 1, 512, 512)
vae_hparams_200x200.enc = (
    # in_channels, out_channels, kernel_size, stride, padding
    ["Conv2d", (3, 32, 3, 2, 1)],
    ["GELU", ()],

    ["Conv2d", (32, 64, 3, 2, 1)],
    ["GELU", ()],

    ["Conv2d", (64, 128, 3, 2, 1)],
    ["GELU", ()],

    ["Conv2d", (128, 128, 3, 2, 1)],
    ["GELU", ()],

    ["Conv2d", (128, 128, 3, 2, 1)],
    ["GELU", ()],

)
vae_hparams_200x200.dec = (
    ["Unflatten", (128, 16, 16)],
    # in_channels, out_channels, kernel_size, stride, padding, output_padding
    ["ConvTranspose2d", (128, 128, 4, 2, 1, 0)],
    ["GELU", ()],

    ["ConvTranspose2d", (128, 128, 4, 2, 1, 0)],
    ["GELU", ()],

    ["ConvTranspose2d", (128, 64, 3, 2, 1, 1)],
    ["GELU", ()],

    ["ConvTranspose2d", (64, 32, 4, 2, 1, 0)],
    ["GELU", ()],

    ["ConvTranspose2d", (32, 1, 4, 2, 1, 0)],
    ["Sigmoid", ()],
)

# Define dictionary assigning actual label names to labels
label_dict = {
    "0": "0_Fine",
    "1": "1_Fine",
    "2": "2_Fine",
    "3": "3_Middle",
    "4": "4_Middle",
    "5": "5_Middle",
    "6": "6_Coarse",
    "7": "7_Coarse",
    "8": "8_Coarse"
}

# Define dictionary with configuration for evaluation metrics
metrics = {
    "Shape Descriptors": "Shape_Factor",
    "Num_images": 500,
    "Skimage_Parameters": {
        "perimeter": "on",
        "area": "on",
        "max_feret": "on",
        "min_feret": "on",
        "orientation": "on",
        "porosity": "on",
        "solidity": "on",
        "curvature_pos": "on",
        "curvature_neg": "on"
    },
    "Image Type": "Micrograph"  # "Micrograph" if micrograph
}


def get_eval_config(args, hparams, device):
    """Workaround function creating a config dict for evaluation function."""
    config = {"Generator": {}, "Discriminator": {}, "get_data_loaders": {}}
    config["Generator"]["nz"] = hparams.z_dim
    config["Discriminator"]["device"] = device
    root_dir, folder = os.path.split(args.path_data)
    config["get_data_loaders"]["dataroot"] = root_dir
    config["get_data_loaders"]["Analysis_File"] = args.path_analysis_file
    config["get_data_loaders"]["image_size"] = hparams.input_size[-1]
    config["get_data_loaders"]["real_size"] = 100
    config["get_data_loaders"]["noise"] = 0
    return config
