import sys
import argparse

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchdyn.core import NeuralODE
from torchdyn.models import CNF as CNFWrapper
from omegaconf import DictConfig

sys.path.append('../')
from datasets.dist import Gaussian, GMM, Funnel, build_augmented_distribution
from datasets.verlet import VerletData
from model.flow import VerletFlow, NonVerletFlow, NonVerletTimeFlow, build_augmented_flow

# DATASETS

# For training 2D CNF's
class ToyDataset(data.Dataset):
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

    def sample(self, N, t=0.0):
        x = self.source.sample(N)
        logp = torch.zeros(N,1).to(x)
        t0 = t * torch.ones(N,1).to(x)
        return torch.cat([logp, t0, x], dim=1)

class VerletDataset(data.Dataset):
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

    def sample(self, N, t=0.0):
        x = self.source.sample(N)
        logp = torch.zeros(N,1).to(x.device)
        t0 = t * torch.ones(N,1).to(x.device)
        # Remove Verlet wrapper
        x = x.get_qp()
        return torch.cat([logp, t0, x], dim=1)

# Based off of FFJORD's CNF
class TorchdynTimeFlow(nn.Module):
    def __init__(self, num_hidden_units):
        super().__init__()
        self.linear1 = nn.Linear(3, num_hidden_units)
        self.linear2 = nn.Linear(num_hidden_units+1, num_hidden_units)
        self.linear3 = nn.Linear(num_hidden_units+1, num_hidden_units)
        self.linear4 = nn.Linear(num_hidden_units+1, 2)

    def forward(self, x):
        t = x[:, :1]
        x = x[:, 1:]
        x = torch.cat([x, t], dim=1)
        x = self.linear1(x)
        x = F.selu(x)
        x = torch.cat([x, t], dim=1)
        x = self.linear2(x)
        x = F.selu(x)
        x = torch.cat([x, t], dim=1)
        x = self.linear3(x)
        x = F.selu(x)
        x = torch.cat([x, t], dim=1)
        dx = self.linear4(x)
        dt = torch.ones_like(t).to(x)
        return torch.cat([dt, dx], dim=1)


# torchdyn-compatible flow
class TorchdynFlow(nn.Module):
    def __init__(self, num_hidden_units):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(3, 25),
            nn.SELU(),
            nn.Linear(25, 40),
            nn.SELU(),
            nn.Linear(40, 25),
            nn.SELU(),
            nn.Linear(25, 2)
        )
    def forward(self, x):
            dt = torch.ones_like(x[:,:1])
            dx = self.f(x)
            return torch.cat([dt, dx], dim=1)

class TorchdynPhaseFlow(nn.Module):
    def __init__(self, phase_flow):
        super().__init__()
        self.flow = phase_flow

    def forward(self, x):
        t = x[:, :1]
        q = x[:, 1:3]
        p = x[:, 3:]
        data = VerletData(q, p, t)
        dq, dp = self.flow.get_flow(data)
        dt = torch.ones_like(t).to(x)
        return torch.cat([dt, dq, dp], dim=1)


# Time-dependent continuous normalizing flow
class CNF(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        self.t_span = torch.linspace(0.0, 1.0, self.args.num_timesteps)

        # Initialize model
        if self.args.use_time:
            self.flow = TorchdynTimeFlow(self.args.num_hidden_units)
        else:
            self.flow = TorchdynFlow(self.args.num_hidden_units)
        self.model = NeuralODE(CNFWrapper(self.flow), sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4)

        # Initialize source, target, train
        self.source = self.build_source(self.args)
        self.target = self.build_target(self.args)
        self.source_set = ToyDataset(self.source, self.args.num_train)
        self.target_set = ToyDataset(self.target, self.args.num_train)

    def build_source(self, args):
        if args.source == 'gaussian':
            # return Gaussian(2.0 * torch.ones(2, device=self.device), torch.tensor([[5.0, 2.0], [2.0, 1.0]], device=self.device))
            return Gaussian(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))
        elif args.source == 'gmm':
            return GMM(nmode=args.source_nmode, device=self.device)
        else:
            raise NotImplementedError

    def build_target(self, args):
        if args.target == 'gaussian':
            return Gaussian(2.0 * torch.ones(2, device=self.device), torch.tensor([[5.0, 2.0], [2.0, 1.0]], device=self.device))
            # return Gaussian(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))
        elif args.target == 'gmm':
            return GMM(nmode=args.target_nmode, device=self.device)
        elif args.target == 'funnel':
            return Funnel(device=self.device, dim=2)
        else:
            raise NotImplementedError

    def setup(self, stage):

        # Move to device
        self.source.to(self.device)
        self.target.to(self.device)
        self.train_losses = []
        
        # Initialize source, target, train
        self.trainloader = data.DataLoader(self.target_set, batch_size=self.args.batch_size, shuffle=True)


    def forward(self, x):
        return self.model(x)

    def compute_data_loss(self, x0):
        _, trajectory = self.model(x0, self.t_span)
        x1 = trajectory[-1]
        flow_logp = x1[:, 0]
        source_logp = self.source.get_density(x1[:, 2:])
        return -torch.mean(source_logp - flow_logp)

    def data_loss(self, N=10000):
        x0 = self.target_set.sample(N, t=0.0)
        return self.compute_data_loss(x0)

    def reverse_kl(self, num_timesteps = 25, N=10000):
        x1 = self.source_set.sample(N, t=1.0)
        source_logp = self.source.get_density(x1[:, 2:])
        t_span = torch.linspace(1.0, 0.0, num_timesteps)
        _, trajectory = self.model(x1, t_span)
        x0 = trajectory[-1] # select last point of solution trajectory
        flow_logp = x0[:,0]
        x0 = x0[:,2:]
        pushforward_logp = source_logp + flow_logp
        target_logp = self.target.get_density(x0)
        return torch.mean(pushforward_logp - target_logp)

    def training_step(self, batch, batch_idx):
        loss = self.compute_data_loss(batch)
        self.train_losses.append(loss)
        return {'loss': loss}

    def on_train_epoch_end(self):
        # Log training loss
        avg_loss = torch.stack(self.train_losses).mean()
        self.train_losses = []
        print(f"Device {self.device} | Epoch {self.current_epoch} | Train Loss: {avg_loss}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return self.trainloader

    # Utility functions 
    def graph(self, trajectory):
        # Plot evolution
        N = trajectory.shape[0]
        fig, axs = plt.subplots(1, N, figsize=(8 * N,8))
        for i in range(N):
            axs[i].hist2d(trajectory[i,:,2], trajectory[i,:,3], bins=100, density=True)
            axs[i].set_xlim(-4, 4)
            axs[i].set_ylim(-4, 4)
        plt.show()      

    @torch.no_grad()
    def graph_marginals(self, num_marginals=5, N=100000):
        X_test = self.source_set.sample(N, t=1.0)
        t_span = torch.linspace(1.0, 0.0, num_marginals)
        t_eval, trajectory = self.model(X_test.cpu(), t_span.cpu())
        trajectory = trajectory.detach().cpu().numpy()
        self.graph(trajectory)

    @torch.no_grad()
    def graph_backwards_marginals(self, num_marginals=5, N=100000):
        X_target = self.target_set.sample(N, t=0.0)
        t_span = torch.linspace(0.0, 1.0, num_marginals)
        t_eval, trajectory = self.model(X_target.cpu(), t_span.cpu())
        trajectory = trajectory.detach().cpu().numpy()
        self.graph(trajectory)

    @staticmethod
    def load_saved(self, path):
        return CNF.load_from_checkpoint(path)

# TODO: Reduce redundancy between CNF and VerletCNF
class PhaseSpaceCNF(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Initialize timespan
        self.t_span = torch.linspace(0.0, 1.0, self.cfg.training.num_timesteps)

        # Initialize model
        flow = build_augmented_flow(self.cfg.flow)
        self.flow = TorchdynPhaseFlow(flow)
        self.model = NeuralODE(CNFWrapper(self.flow), sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4)

        # Initialize source, target, train
        self.source = build_augmented_distribution(self.cfg.source, self.device, 1.0)
        self.target = build_augmented_distribution(self.cfg.target, self.device, 0.0)
        self.source_set = VerletDataset(self.source, self.cfg.training.num_train)
        self.target_set = VerletDataset(self.target, self.cfg.training.num_train)

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

    def compute_data_loss(self, x0):
        _, trajectory = self.model(x0, self.t_span)
        x1 = trajectory[-1]
        flow_logp = x1[:, 0]
        x1 = VerletData.from_qp(x1[:, 2:], t=1.0)
        source_logp = self.source.get_density(x1)
        return -torch.mean(source_logp - flow_logp)

    def data_loss(self, N=10000):
        x0 = self.target_set.sample(N, t=0.0)
        return self.compute_data_loss(x0)

    def reverse_kl(self, num_timesteps = 25, N=10000):
        x1 = self.source_set.sample(N, t=1.0)
        x1_data = VerletData.from_qp(x1[:,2:], t=1.0)
        source_logp = self.source.get_density(x1_data)
        t_span = torch.linspace(1.0, 0.0, num_timesteps)
        _, trajectory = self.model(x1, t_span)
        x0 = trajectory[-1] # select last point of solution trajectory
        flow_logp = x0[:,0]
        x0 = VerletData.from_qp(x0[:,2:], 1.0)
        pushforward_logp = source_logp + flow_logp
        target_logp = self.target.get_density(x0)
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
        X_test = self.source_set.sample(N, t=1.0)
        t_span = torch.linspace(1.0, 0.0, num_marginals)
        t_eval, trajectory = self.model(X_test.cpu(), t_span.cpu())
        trajectory = trajectory.detach().cpu().numpy()
        self.graph(trajectory)

    def graph_backwards_marginals(self, num_marginals=5, N=10000):
        X_target = self.target_set.sample(N, t=0.0)
        t_span = torch.linspace(0.0, 1.0, num_marginals)
        t_eval, trajectory = self.model(X_target.cpu(), t_span.cpu())
        trajectory = trajectory.detach().cpu().numpy()
        self.graph(trajectory)

    @staticmethod
    def load_saved(self, path):
        return PhaseSpaceCNF.load_from_checkpoint(path)
