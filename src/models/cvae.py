"""Module with classes to define / train Variational Autoencoder"""

# Import third party dependencies
import torch
import torch.nn as nn

# Import local dependencies
from src.models.base_vae import BaseVAE


class Flatten(nn.Module):
    """A torch.nn.Module to flatten the hidden representation in the
    Variational Autoencoder after encoding the input data."""

    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    """A torch.nn.Module to unflatten the latent representation before passing
    it to the decoder part of the Variational Autoencoder."""

    def __init__(self, channels, width, height):
        super(Unflatten, self).__init__()
        self.channels = channels
        self.width = width
        self.height = height

    def forward(self, input):
        return input.view(
            input.size(0), self.channels, self.width, self.height
        )


def get_layer(layer_config):
    """Get Pytorch layer based on specified config."""
    layer_string = layer_config[0]
    layer_hparams = layer_config[1]
    if layer_string == "Unflatten":
        layer = Unflatten(*layer_hparams)
    else:
        layer = getattr(nn, layer_string)(*layer_hparams)
    return layer


class Encoder(nn.Module):
    """Convolutional Encoder"""

    def __init__(self, hparams, device):
        super(Encoder, self).__init__()
        self.num_classes = hparams.num_classes
        self.device = device
        self.img_size = 512

        self.embed_1 = nn.Embedding(
            3,
            hparams.input_size[-1] * hparams.input_size[-1]
        )
        self.embed_2 = nn.Embedding(
            3,
            hparams.input_size[-1] * hparams.input_size[-1]
        )

        self.layers = nn.Sequential(
            *[get_layer(layer) for layer in hparams.enc]
        )

        self.flatten = Flatten()

        self.fc_mu = nn.Linear(hparams.h_dim, hparams.z_dim)
        self.fc_log_variance = nn.Linear(hparams.h_dim, hparams.z_dim)

    def forward(self, x, c):

        # Embed labels
        label_p = torch.div(c, 3, rounding_mode='floor').to(self.device).int()
        label_d = torch.remainder(c, 3).to(self.device).int()
        embedding_1 = self.embed_1(
            label_p
        ).view(
            label_p.size()[0], 1, self.img_size, self.img_size
        ).to(self.device)
        embedding_2 = self.embed_2(
            label_d
        ).view(
            label_d.size()[0], 1, self.img_size, self.img_size
        ).to(self.device)
        x = torch.cat([x.to(self.device), embedding_1, embedding_2], dim=1)

        # Forward pass through Sequential block
        x = self.layers(x)

        # Flatten
        x = self.flatten(x)

        mu = self.fc_mu(x)
        log_variance = self.fc_log_variance(x)

        return mu, log_variance


class Decoder(nn.Module):
    """Convolutional Decoder"""

    def __init__(self, hparams, device):
        super(Decoder, self).__init__()
        self.num_classes = hparams.num_classes
        self.device = device

        self.embed_1 = nn.Embedding(3, int(hparams.z_dim/2))
        self.embed_2 = nn.Embedding(3, int(hparams.z_dim/2))

        self.fc = nn.Linear(hparams.z_dim + hparams.z_dim, hparams.h_dim)

        self.unflatten = get_layer(hparams.dec[0])

        self.layers = nn.Sequential(
            *[get_layer(layer) for layer in hparams.dec[1:]]
        )

    def forward(self, x, c):

        label_p = torch.div(c, 3, rounding_mode='floor').to(self.device).int()
        label_d = torch.remainder(c, 3).to(self.device).int()
        embedding_1 = self.embed_1(label_p).to(self.device)
        embedding_2 = self.embed_2(label_d).to(self.device)

        x = torch.cat((x.to(self.device), embedding_1, embedding_2), dim=1)

        x = self.fc(x)
        x = self.unflatten(x)
        x = self.layers(x)

        return x


class VariationalAutoencoder(BaseVAE):
    """A class based on torch.nn.Module implementing a Variational Autoencoder

    Parameters
    ----------
    hparams : config.Hyperparameters
        Dictionary-like object with hyperparameters that can be accessed via
        point-operator (e.g. hparams.z_dim).
    device : torch.device
        Torch device (GPU/CPU)
    """

    def __init__(self, hparams, device):
        super(VariationalAutoencoder, self).__init__(hparams)

        self.device = device
        self.z_dim = hparams.z_dim
        self.num_classes = hparams.num_classes

        # Encoder
        self.encoder = Encoder(hparams, device=device)

        # Decoder
        self.decoder = Decoder(hparams, device=device)

    def reparametrize(self, mu, log_sigma):
        """Do reparametrization trick and draw sample."""
        if self.training:
            sigma = torch.exp(log_sigma)
            epsilon = torch.randn_like(sigma)
            sample = mu + epsilon * sigma
        else:
            sample = mu

        return sample

    def forward(self, x, c):
        """Forward pass data."""
        # Encode input into hidden state
        mu, log_sigma = self.encoder(x, c)

        # Apply reparametrization trick to sample latent representation
        z = self.reparametrize(mu, log_sigma)

        # Reconstruct input by decoding latent representation z
        rec = self.decoder(z, c)

        # Get KL divergence
        kld = self._get_kld(mu, log_sigma)

        return rec, kld

    def generate_samples(self, n_samples, c):
        """Generate *n_samples* samples for condition *c*."""
        # Set model to eval mode
        self.eval()

        # Draw samples in latent space and decode to original space
        z = torch.randn(n_samples, self.z_dim)
        c = torch.Tensor(n_samples*[c]).long()
        return self.decoder(z, c).cpu().detach().numpy()
