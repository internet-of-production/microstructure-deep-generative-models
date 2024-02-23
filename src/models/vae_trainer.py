
"""Trainer class to fit VAEs. Based on [1].

[1] https://github.com/EugenHotaj/pytorch-generative/
"""

# Standard library imports
import datetime
import glob
import os
import re
from typing import Callable, Tuple

# Import third party dependencies
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
from torch.utils import tensorboard
from tqdm import tqdm


class Trainer:
    """Class to train VAEs."""

    def __init__(
        self,
        model: torch.nn.Module,
        input_shape: Tuple,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: torch.device,
        log_dir: str,
        beta: int = 1
    ):
        """Instantiate object of Trainer class.

        Parameters
        ----------
        model : torch.nn.Module
            A VAE that inherited from torch.nn.Module.
        input_shape: list-like
            Sequence of integers defining the input shape
        optimizer : torch.optim.Optimizer
            A torch optimizer.
        loss_fn : Callable
            Function that computes an ELBO loss.
        device : torch.device
            Torch device (cpu, gpu, mps).
        log_dir : str
            Path to directory where Tensorboard logs are saved.
        beta : int, optional
            Beta factor for KL divergence in ELBO loss, by default 1.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.beta = beta
        self.device = device
        self.log_dir = log_dir
        self._step = 0
        self._epoch = 0
        self._loss = 0

        self._summary_writer = tensorboard.SummaryWriter(self.log_dir)
        self._summary_writer.add_graph(
            model,
            [
                torch.zeros(input_shape),
                torch.zeros(input_shape[0], dtype=torch.int64)
            ]
        )

    def _update_tensorboard(self, metrics: dict):
        """Add metrics as scalar valus to Tensorboard.

        Parameters
        ----------
        metrics : dict
            Dictionary with metric names (keys) and values.
        """
        for key, value in metrics.items():
            self._summary_writer.add_scalar(
                f"metrics/{key}",
                value,
                self._epoch
            )

    def _add_hparams_to_tensorboard(self, metrics: dict):
        """Add hyperparameters to Tensorboard log.

        Parameters
        ----------
        metrics : dict
            Dictionary with evaluation metrics.
        """
        self.model.hparams["epoch"] = int(self._epoch)
        self._summary_writer.add_hparams(
            self.model.hparams,
            {"hparams/"+k: v for (k, v) in metrics.items()},
            run_name=self._run_name
        )

    def _add_samples_to_tensorboard_images(self, n_samples: int = 32):
        """Generate samples with VAE and add samples as images to Tensorboard.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to be generated, by default 32
        """
        for c in range(self.model.num_classes):
            samples = self.model.generate_samples(n_samples, c)

            self._summary_writer.add_images(
                f"images/samples/label_{c}",
                samples,
                self._epoch,
                dataformats="NCHW"
            )

    def _add_recon_to_tensorboard_images(
        self, x: torch.Tensor,
        x_recon: torch.Tensor
    ):
        """Create plots with input data and reconstructions.

        Parameters
        ----------
        x : torch.Tensor
            Original data with shape (batch_size, 1, width, height).
        x_recon : torch.Tensor
            Reconstructed data with shape (batch_size, 1, width, height).
        """
        images = []

        for (original, recon) in zip(x, x_recon):
            # make a Figure and attach it to a canvas.
            fig = Figure(figsize=(5, 4), dpi=100)
            canvas = FigureCanvasAgg(fig)

            # Do some plotting here
            ax1 = fig.add_subplot(121)
            ax1.imshow(original.detach().cpu().numpy().squeeze(), cmap="gray")
            ax1.set_title("Original")
            ax1.set_axis_off()
            ax2 = fig.add_subplot(122)
            ax2.imshow(recon.detach().cpu().numpy().squeeze(), cmap="gray")
            ax2.set_title("Reconstruction")
            ax2.set_axis_off()

            # Retrieve a view on the renderer buffer
            canvas.draw()
            buf = canvas.buffer_rgba()
            # convert to a NumPy array
            img = np.asarray(buf)
            images.append(img[:, :, :-1])

        images = np.array(images)
        self._summary_writer.add_images(
            "images/reconstructions",
            images,
            self._epoch,
            dataformats="NHWC"
        )

    def _get_last_epoch(self):
        """Get the latest epoch from file names of checkpoints.

        Check all files in log directory for Tensorboard for files with name
        that matches the pattern "trainer_state_[0-9]*.ckpt". Get and return
        the latest epoch from those file names.

        Returns
        -------
        int
           Latest epoch

        Raises
        ------
        FileNotFoundError
            Raise error if no checkpoint with a name matching
            "trainer_state_[0-9]*.ckpt" is found in the log directory.
        """
        files = glob.glob(self._path("trainer_state_[0-9]*.ckpt"))
        epochs = sorted([int(re.findall(r"\d+", f)[-1]) for f in files])
        if not epochs:
            raise FileNotFoundError(f"No checkpoints found in {self.log_dir}.")
        print(f"Found {len(epochs)} saved checkpoints.")
        return epochs[-1]

    def _path(self, file_name: str):
        """Return path joining Tensorboard log directory and file name.

        Parameters
        ----------
        file_name : str
            Name of a file

        Returns
        -------
        str
            Path that joins path of log directory and the given file name.
        """
        return os.path.join(self.log_dir, file_name)

    def _save_checkpoint(self):
        """Save checkpoint including model, optimizer, epoch and optimization
        step.
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
            "epoch": self._epoch,
            "run_name": self._run_name
        }
        torch.save(checkpoint, self._path(f"trainer_state_{self._epoch}.ckpt"))

    def _restore_checkpoint(self):
        """Load latest checkpoint to continue training."""
        self._epoch = self._get_last_epoch()
        checkpoint = f"trainer_state_{self._epoch}.ckpt"
        print(f"Continuing training from checkpoint {checkpoint}.")
        checkpoint = torch.load(
            self._path(checkpoint),
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._step = checkpoint["step"]
        self._epoch = checkpoint["epoch"]
        self._run_name = checkpoint["run_name"]

        # checkpoint
        self._summary_writer.close()
        self._summary_writer = tensorboard.SummaryWriter(
            self.log_dir,
            max_queue=10,
            purge_step=self._step
        )

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        continue_training: bool = True,
    ):
        """Fit model.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader with training dataset.
        num_epochs : int
            Number of training epochs.
        continue_training : bool
            If True try to load checkpoint of model and continue training, by
            default True.
        """
        if continue_training:
            try:
                self._restore_checkpoint()
            except FileNotFoundError:
                self._run_name = datetime.datetime.now().strftime(
                    "%Y%m%d-%H%M%S"
                )

        for e in range(self._epoch, num_epochs):
            self.model.train()

            progress_bar = tqdm(
                train_loader,
                desc=f"{self._epoch+1}/{num_epochs}"
            )
            for x, c in progress_bar:
                x = x.to(self.device)
                c = c.to(self.device)

                self.optimizer.zero_grad()

                preds = self.model(x, c)

                metrics = self.loss_fn(x, c, preds)
                loss = metrics["loss"]

                self._loss = metrics["loss"].item()
                self._kld = metrics["kld"].item()
                self._re = metrics["recon_loss"].item()
                progress_bar.set_description(
                    f"{e+1}/{num_epochs} - "
                    f"Loss: {self._loss} | "
                    f"RE: {self._re} | "
                    f"KLD {self._kld}"
                )

                loss.backward()
                self.optimizer.step()

                self._step += 1

            # Update Tensorboard
            self._update_tensorboard(metrics)

            self._epoch += 1
            # Save images every 50 epochs
            if not self._epoch % 50:
                self._add_recon_to_tensorboard_images(x, preds[0])
                self._add_samples_to_tensorboard_images()
                self._save_checkpoint()
            self._add_hparams_to_tensorboard(metrics)
