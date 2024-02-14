import sys
import argparse

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchdyn.core import NeuralODE
# from torchdyn.models import CNF as CNFWrapper
from model.torchdyn import CNF as CNFWrapper
from omegaconf import DictConfig

sys.path.append('../')
from datasets.dist import Gaussian, GMM, Funnel, build_augmented_distribution
from datasets.aug_data import AugmentedData
from model.flow import TorchdynAugmentedFlowWrapper, build_augmented_flow
from model.wrapper import AugmentedWrapper

# DATASETS


class TorchdynAugmentedDataset(data.Dataset):
    def __init__(self, source, length):
        self.length = length
        self.source = source

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.sample(1).squeeze(0)

    def to(self, device):
        self.source.to(device)
        return self

    def sample(self, N):
        x = self.source.sample(N)
        logp = torch.zeros(N,1).to(x.device)
        t0 = self.source.t * torch.ones(N,1).to(x.device)
        # Remove Verlet wrapper
        x = x.get_qp()
        return torch.cat([logp, t0, x], dim=1)


# TODO: Reduce redundancy between CNF and VerletCNF
class AugmentedCNF(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Initialize timespan spanning target at t=1.0 to source at t=0.0
        self.t_span = torch.linspace(1.0, 0.0, self.cfg.training.num_timesteps)

        # Initialize source, target, train
        self.source = build_augmented_distribution(self.cfg.source, self.device, 0.0)
        self.target = build_augmented_distribution(self.cfg.target, self.device, 1.0)
        self.source_set = TorchdynAugmentedDataset(self.source, self.cfg.training.num_train)
        self.target_set = TorchdynAugmentedDataset(self.target, self.cfg.training.num_train)
        
        # Initialize model
        flow = build_augmented_flow(self.cfg.flow, self.target)
        self.flow = TorchdynAugmentedFlowWrapper(flow)
        self.model = NeuralODE(CNFWrapper(self.flow), sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4)

    def align_devices(self):
        self.source.to(self.device)
        self.target.to(self.device)
        self.model.to(self.device)


    def setup(self, stage):
        # Update devices
        self.source.to(self.device)
        self.target.to(self.device)
        self.train_losses = []
        
        # Initialize source, target, train
        self.train_loader = data.DataLoader(self.target_set, batch_size=self.cfg.training.batch_size, shuffle=True)
        self.val_loader = data.DataLoader(self.target_set, batch_size=self.cfg.training.batch_size, shuffle=True)

    def forward(self, x):
        return self.model(x)

    def compute_data_loss(self, x1):
        _, trajectory = self.model(x1, self.t_span)
        x0 = trajectory[-1]
        flow_logp = x0[:, 0]
        x0 = AugmentedData.from_qp(x0[:, 2:], t=0.0)
        source_logp = self.source.get_log_density(x0)
        print(f'Source: {-torch.mean(source_logp)}, flow: {-torch.mean(flow_logp)}')
        return -torch.mean(source_logp - flow_logp)

    def data_loss(self, N=10000):
        x1 = self.target_set.sample(N)
        return self.compute_data_loss(x1)

    def reverse_kl(self, num_timesteps = 25, N=10000):
        x0 = self.source_set.sample(N)
        x0_data = AugmentedData.from_qp(x0[:,2:], t=0.0)
        source_logp = self.source.get_log_density(x0_data)
        t_span = torch.linspace(0.0, 1.0, num_timesteps)
        _, trajectory = self.model(x0, t_span)
        x1 = trajectory[-1] # select last point of solution trajectory
        flow_logp = x1[:,0]
        x1 = AugmentedData.from_qp(x1[:,2:], t=1.0)
        pushforward_logp = source_logp + flow_logp
        target_logp = self.target.get_log_density(x1)
        return torch.mean(pushforward_logp - target_logp)

    def training_step(self, batch, batch_idx):
        loss = self.compute_data_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_losses.append(loss)
        return {'loss': loss}

    def on_train_epoch_end(self):
        # Log training loss
        avg_loss = torch.stack(self.train_losses).mean()
        self.train_losses = []
        print(f"Device {self.device} | Epoch {self.current_epoch} | Train Loss: {avg_loss}")

    def validation_step(self, batch, batch_idx):
        loss = self.compute_data_loss(batch)
        self.log("val_loss", loss)
        return {'val_loss': loss}


    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    # Utility functions
    def graph(self, trajectory):
        # Plot evolution
        N = trajectory.shape[0]
        fig, axs = plt.subplots(1, N, figsize=(8 * N,8))
        for i in range(N):
            axs[i].hist2d(trajectory[i,:,2], trajectory[i,:,3], bins=100, density=True)
        plt.show()

    def graph_marginals(self, num_marginals=5, N=10000):
        X_test = self.source_set.sample(N)
        t_span = torch.linspace(0.0, 1.0, num_marginals)
        t_eval, trajectory = self.model(X_test.to(self.device), t_span.to(self.device))
        trajectory = trajectory.detach().cpu().numpy()
        self.graph(trajectory)

    def graph_backwards_marginals(self, num_marginals=5, N=10000):
        X_target = self.target_set.sample(N)
        t_span = torch.linspace(1.0, 0.0, num_marginals)
        t_eval, trajectory = self.model(X_target.to(self.device), t_span.to(self.device))
        trajectory = trajectory.detach().cpu().numpy()
        self.graph(trajectory)

    def export_to_wrapper(self):
        return AugmentedWrapper(self.source, self.target, self.flow.flow)

    @staticmethod
    def load_saved(self, path):
        return AugmentedCNF.load_from_checkpoint(path)
