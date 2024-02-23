"""Module with utility functions."""

# Import standard library imports
import json
import os

# Import third party dependencies
import torch


def create_path(path):
    """Create path if it not yet exists

    :param path: path to be created
    :type path: str
    """

    if not os.path.exists(path):
        os.makedirs(path)


def load_model(path, device="cpu"):
    model = torch.load(path, map_location=torch.device(device))
    return model


def save_model_and_hparams(path, model, hparams):
    create_path(path)
    torch.save(model, os.path.join(path, "vae.pt"))
    hparams_file = open(os.path.join(path, "hparams.json"), "w")
    hparams_file = json.dump(hparams, hparams_file, indent=4)
    print(f"Model and hparams saved at {path}.")
