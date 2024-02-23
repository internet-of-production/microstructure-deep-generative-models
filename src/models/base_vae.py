import torch
import torch.nn as nn


class BaseVAE(nn.Module):

    def __init__(self, hparams):
        super(BaseVAE, self).__init__()
        self.hparams = {
            "learn_rate": hparams.learn_rate,
            "beta": hparams.beta,
            "shuffle_data": hparams.shuffle_data,
            "latent_dim": hparams.z_dim
        }

    def _get_kld(self, latent_mean, latent_log_var):
        kld = torch.mean(
            -0.5 * torch.sum(
                1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp(),
                dim=1
            )
        )
        return kld

    def generate_samples(self, n_samples):
        # Set model to eval mode
        self.eval()

        # Draw sample
        p = torch.distributions.Normal(
            torch.zeros_like(n_samples, self.z_dim),
            torch.ones_like(n_samples, self.z_dim)
        )
        z = p.rsample()

        return self.decoder(z).cpu().detach().numpy()


class ELBO:

    def __init__(self, loss_fn, beta, warm_up_period=0):
        self.beta = beta
        self.loss_fn = loss_fn
        self.warm_up_period = warm_up_period

    def __call__(self, x, _, preds, epoch=None):
        if epoch:
            fraction = max(epoch / self.warm_up_period, 1)
        else:
            fraction = 1
        x_pred, kld = preds
        recon_loss = self.loss_fn(x_pred, x, reduction="none")
        recon_loss = recon_loss.sum(dim=(1, 2, 3))
        elbo = recon_loss + fraction * self.beta * kld
        return {
            "recon_loss": recon_loss.mean(),
            "kld": kld.mean(),
            "loss": elbo.mean()
        }
