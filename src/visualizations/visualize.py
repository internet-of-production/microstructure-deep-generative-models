"""Module with functions to create plots."""

# Standard library imports
import os
from typing import Union

# Import third party dependencies
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_pore(
    pore,
    path: str = None,
    title: str = None,
    plot_axes: bool = False,
    bbox_inches=None
):
    """Plot pore as image using Pyplots imshow.

    Parameters
    ----------
    pore : array-like or PIL image
        Pore to be plotted with shape (W, H, 3) or (W, H), where W is width and
        H is height.
    path : str, optional
        Path under which, the figure is saved. If None, the figure is only
        shown but not saved. Defaults to None.
    title : str, optional
        Title used in plot, by default None.
    plot_axes : bool, optional
        Determines whether to keep axes or turn them off, by default False.
    bbox_inches : str, optional
         cf. Pyplot documentation: 'If 'tight', try to figure out the tight
         bbox of the figure.'
    """
    # Plot pore
    my_dpi = 10
    plt.figure(frameon=False, figsize=(665/my_dpi, 665/my_dpi), dpi=my_dpi)
    plt.imshow(pore, cmap="gray")

    if not plot_axes:
        plt.axis("off")
        plt.autoscale(tight=True)

    if title:
        plt.title(title)

    # Save figure to file, if path is specified
    if path:
        plt.savefig(path, bbox_inches="tight", pad_inches=0.0, dpi=my_dpi)
        plt.close()
    else:
        plt.show()


def plot_original_vs_reconstruction(x, rec, path=None):
    """Create figure with original image and reconstructed image side by side.

    Parameters
    ----------
    x : array-like
        Original image with shape (W, H, 3) or (W, H), where W is width and H
        is height.
    rec : array-like
        Reconstucted image with shape (W, H, 3) or (W, H), where W is width and
        H is height.
    path : str, optional
        Path under which, the figure is saved. If None, the figure is only
        shown but not saved. Defaults to None.
    """
    # Create axes
    fig, axes = plt.subplots(ncols=2)

    # Plot original pore and reconstruction
    axes[0].imshow(x, cmap="gray")
    axes[1].imshow(rec, cmap="gray")

    # Add titles
    axes[0].set_title("Original")
    axes[1].set_title("Reconstruction")

    # Save figure to file, if path is specified
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_n_vae_samples(
    vae,
    n: int,
    z_dim: int,
    c: torch.Tensor = None,
    path: str = None,
    device: Union[str, torch.device] = "cpu"
):
    """Generate n samples with VAE and plot samples.

    Parameters
    ----------
    vae : src.models.VariationalAutoencoder
        a Variational Autoencoder
    n : int
        Number of samples to be generated.
    z_dim : int
        Dimensionality of latent space.
    c : torch.Tensor, optional
        Condition for conditional VAE, by default None.
    path : str, optional
        Path to folder in which, figures are saved. If None, the figures only
        shown but not saved. Defaults to None
    device : Union[str, torch.device], optional
        torch device (GPU/CPU). Specify to make sure that vae and c are on the
        same device. Defaults to cpu.
    """
    vae.eval()

    with torch.no_grad():
        z = torch.randn(n, z_dim, device=device)
        c = torch.Tensor(n*[c]).long()
        vae = vae.to(device)
        vae.device = device

        # Generate n samples
        if c is None:
            samples = vae.decoder(z)
        else:
            samples = vae.decoder(z, c)

        for i, sample in enumerate(samples):
            if path:
                # Plot sample and save figure
                file_path = os.path.join(path, f"{i}.png")
                plot_pore(
                  np.squeeze(sample.cpu()),
                  file_path,
                  title=None,
                  plot_axes=False,
                  bbox_inches="tight"
                )
            else:
                # Plot sample and show figure if path is None
                plot_pore(np.squeeze(sample.cpu()), title=f"Condition: {c}")


def plot_n_vae_reconstructions(
    vae,
    n: int,
    data_loader: torch.utils.data.DataLoader,
    device: Union[str, torch.device],
    path: str = None
):
    """Reconstruct n input images and plot reconstructions versus originals.


    Parameters
    ----------
    vae : src.models.VariationalAutoencoder
        A Variational Autoencoder.
    n : int
        Number of samples to be reconstructed.
    z_dim : int
        Dimensionality of latent space.
    c : torch.Tensor, optional
        Condition for conditional VAE, by default None.
    path : str, optional
        Path to folder in which, figures are saved. If None, the figures only
        shown but not saved. Defaults to None
    device : Union[str, torch.device], optional
        torch device (GPU/CPU). Specify to make sure that vae and c are on the
        same device. Defaults to cpu.
    """

    # Create variable to track number of plotted reconstructions
    i = 0

    for x, c in iter(data_loader):
        # Break loop if maximum number of plots is reached
        if i >= n:
            break

        # Send data to device
        x = x.to(device)
        c = c.to(device)

        # Reconstruct input image with VAE
        rec, _, _ = vae(x, c)

        # Transform reconstruciton to numpy array and remove axes of length 1
        rec = np.squeeze(rec.detach().cpu().numpy())
        x = np.squeeze(x.cpu())

        # Iterate over pairs of original images and reconstructions
        for o, r in zip(x, rec):

            # Break loop if maximum number of plots is reached
            if i >= n:
                break

            if path:
                # Plot reconstruction and save figure
                file_path = os.path.join(path, f"{i}.png")
                plot_original_vs_reconstruction(o, r, file_path)
            else:
                # Plot reconstruction and show figure if path is None
                plot_original_vs_reconstruction(o, r)

            i += 1
