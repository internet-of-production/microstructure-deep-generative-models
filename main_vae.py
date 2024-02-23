# Import standard library dependencies
import argparse
import os

# Import third party dependencies
import torch
import torch.nn.functional as F
# from torchinfo import summary

# Import local dependencies
import config
import src.utils as utils
from src.data.load_data_VAE import get_data_loader
from src.data.metrics.analyze_metrics import analyze_samples
from src.models.cvae import VariationalAutoencoder
from src.models.vae_trainer import Trainer
from src.models.base_vae import ELBO
from src.visualizations.visualize import plot_n_vae_samples
# from src.visualizations.visualize import plot_n_vae_reconstructions
from src.data.metrics.Analyze2Point import Analyze2Point


class Experiment:

    def __init__(self, config, hparams, args, device):
        self.config = config
        self.hparams = hparams
        self.args = args
        self.device = device

    def _get_data(self):
        return get_data_loader(
            self.args.path_data,
            self.hparams.batch_size,
            self.hparams.shuffle_data
        )

    def _get_loss_fn(self):
        return ELBO(
            F.binary_cross_entropy,
            self.hparams.beta,
            self.hparams.warm_up_period
        )

    def _get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.hparams.learn_rate)

    def _get_model(self):
        return VariationalAutoencoder(self.hparams, device=self.device)

    def _get_metrics(self, epoch_count, experiment_id, vae):

        for c in range(self.hparams.num_classes):
            # Generate samples with VAE and save plots of samples
            path = os.path.join(
                self.args.path_figures,
                experiment_id,
                "generated",
                f"Epoche_{epoch_count}",
                f"Label_{c}"
            )
            utils.create_path(path)
            n = 20  # Number of plots to be generated
            plot_n_vae_samples(
                vae=vae,
                z_dim=vae.z_dim,
                c=c,
                n=n,
                path=path,
                device=self.device
            )
            print(
                f"Exemplary generated samples for condition {c} saved at "
                f"{path}."
            )

            # Analyze generated samples
            print("Analyze generated samples for condition:", c)
            configdata = config.get_eval_config(
                args,
                hparams,
                self.args.device
            )
            analyze_samples(
                vae.decoder,
                configdata=configdata,
                metrics=config.metrics,
                Resultfolder=os.path.join(args.path_results, experiment_id),
                labeldic=config.label_dict,
                label=c,
                currentepoch=epoch_count,
            )
            Analyze2Point(
                configdata,
                c,
                config.label_dict,
                os.path.join(args.path_results, experiment_id),
                epoch_count
            )

    def run(self, run_id: str) -> torch.nn.Module:
        """Start experiment run.

        If checkpoints exist for the given run_id, training will be continued
        from the latest checkpoint.

        Parameters
        ----------
        run_id : str
            Name of experiment used to save logs and results results.

        Returns
        -------
        torch.nn.Module
            The fitted model.
        """
        train_loader = self._get_data()
        model = self._get_model()
        # model.summary(self.hparams.input_size, self.hparams.batch_size)
        optimizer = self._get_optimizer(model)
        loss_fn = self._get_loss_fn()

        # Fit model
        trainer = Trainer(
            model=model,
            input_shape=self.hparams.input_size,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=self.device,
            log_dir=os.path.join(self.args.path_log_dir, run_id),
            beta=self.hparams.beta
        )
        trainer.fit(train_loader, self.hparams.num_epochs)

        # Compute evaluation metrics
        self._get_metrics(self.hparams.num_epochs, run_id, model)

        return model


if __name__ == "__main__":

    EXPERIMENT_ID = "ENTER_RUN_NAME_HERE"

    # Instantiate parser with configuration data paths
    # Parametrization of models is done in config.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str)
    parser.add_argument("--path_figures", type=str)
    parser.add_argument("--path_model", type=str)
    parser.add_argument("--path_results", type=str)
    parser.add_argument("--path_analysis_file", type=str)
    parser.add_argument("--path_log_dir", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    # Get hyperparameters from config file (comment/uncomment for respective
    # microgrpah size)
    hparams = config.vae_hparams_200x200
    # hparams = config.vae_hparams_100x100

    # Run experiment
    experiment = Experiment(
        config,
        hparams,
        args,
        torch.device(args.device)
    )
    experiment.run(EXPERIMENT_ID)
