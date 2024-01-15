import sys
import argparse

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
import pytorch_lightning as pl

from torchdyn.core import NeuralODE
from torchdyn.models import CNF as CNFWrapper

sys.path.append('../')
from datasets.dist import Gaussian, GMM, VerletGMM, VerletGaussian, VerletFunnel
from datasets.verlet import VerletData
from model.flow import VerletFlow


def parse_cnf_args(manual_args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hidden_units', type=int, default=16)
    parser.add_argument('--num_timesteps', type=int, default=25)
    # Source
    parser.add_argument('--source', type=str, default='gaussian')
    parser.add_argument('--source_nmode', type=int, default=2)
    # Target
    parser.add_argument('--target', type=str, default='gmm')
    parser.add_argument('--target_nmode', type=int, default=3)
    args = (
        parser.parse_args() if manual_args is None else parser.parse_args(manual_args)
    )
    return args

def parse_verlet_cnf_args(manual_args = None):
    parser = argparse.ArgumentParser()
    # Flow
    parser.add_argument('--num_vp_hidden', type=int, default=16)
    parser.add_argument('--num_vp_layers', type=int, default=3)
    parser.add_argument('--num_nvp_hidden', type=int, default=16)
    parser.add_argument('--num_nvp_layers', type=int, default=3)
    # Integrator
    parser.add_argument('--num_timesteps', type=int, default=25)
    # Source
    parser.add_argument('--source', type=str, default='gaussian')
    parser.add_argument('--source_nmode', type=int, default=2)
    # Target
    parser.add_argument('--target', type=str, default='gmm')
    parser.add_argument('--target_nmode', type=int, default=2)
    args = (
        parser.parse_args() if manual_args is None else parser.parse_args(manual_args)
    )
    return args

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

    def sample(self, N, t=0.0):
        x = self.source.sample(N)
        logp = self.source.get_density(x)[:,None]
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

    def sample(self, N, t=0.0):
        x = self.source.sample(N)
        logp = self.source.get_density(x)[:,None]
        t0 = t * torch.ones(N,1).to(x.device)
        # Remove Verlet wrapper
        x = x.get_qp()
        return torch.cat([logp, t0, x], dim=1)

# torchdyn-compatible flow
class TorchdynFlow(nn.Module):
    def __init__(self, num_hidden_units):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(3, num_hidden_units),
            nn.SELU(),
            nn.Linear(num_hidden_units, num_hidden_units),
            nn.SELU(),
            nn.Linear(num_hidden_units, num_hidden_units),
            nn.SELU(),
            nn.Linear(num_hidden_units, 2)
        )
    def forward(self, x):
            dt = torch.ones_like(x[:,:1])
            dx = self.f(x)
            return torch.cat([dt, dx], dim=1)

class TorchdynVerletFlow(nn.Module):
    def __init__(self, verlet_flow):
        super().__init__()
        self.flow = verlet_flow

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

    def build_source(self, args):
        if args.source == 'gaussian':
            return Gaussian(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))
        elif args.source == 'gmm':
            return GMM(nmode=args.source_nmode, device=self.device)
        else:
            raise NotImplementedError

    def build_target(self, args):
        if args.target == 'gaussian':
            return Gaussian(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))
        elif args.target == 'gmm':
            return GMM(nmode=args.target_nmode, device=self.device)
        else:
            raise NotImplementedError

    def setup(self, stage):
        self.t_span = torch.linspace(0.0, 1.0, self.args.num_timesteps)

        # Initialize model
        self.flow = TorchdynFlow(self.args.num_hidden_units)
        self.model = NeuralODE(CNFWrapper(self.flow), sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4)
        
        # Initialize source, target, train
        self.source = self.build_source(self.args)
        self.target = self.build_target(self.args)
        self.source_set = ToyDataset(self.source, 1000)
        self.target_set = ToyDataset(self.target, 1000)
        self.trainloader = data.DataLoader(self.source_set, batch_size=250, shuffle=True)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x0 = batch
        t_eval, trajectory = self.model(x0, self.t_span)
        x1 = trajectory[-1] # select last point of solution trajectory
        pushforward_logp = x1[:,0]
        pushforward_x = x1[:,2:]
        target_logp = self.target.get_density(pushforward_x)
        loss = torch.mean(pushforward_logp - target_logp)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return self.trainloader

    # Utility functions 
    def graph(self, trajectory):
        # Plot evolution
        N = len(self.t_span)
        fig, axs = plt.subplots(1, N, figsize=(8 * N,8))
        for i in range(N):
            axs[i].hist2d(trajectory[i,:,2], trajectory[i,:,3], bins=100, density=True)
        plt.show()      

    def graph_marginals(self, N):
        X_test = self.source_set.sample(N)
        t_span = self.t_span
        t_eval, trajectory = self.model(X_test.cpu(), t_span.cpu())
        trajectory = trajectory.detach().cpu().numpy()
        self.graph(trajectory)

    def graph_backwards_marginals(self, N):
        X_target = self.target_set.sample(N, t=1.0)
        t_span = torch.flip(self.t_span, dims=(0,))
        t_eval, trajectory = self.model(X_target.cpu(), t_span.cpu())
        trajectory = trajectory.detach().cpu().numpy()
        self.graph(trajectory)

    @staticmethod
    def load_saved(self, path):
        return CNF.load_from_checkpoint(path)



# TODO: Reduce redundancy between CNF and VerletCNF
class CoupledCNF(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

    def build_source(self, args):
        if args.source == 'gaussian':
            q_dist = Gaussian(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))
            p_dist = Gaussian(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))
            return VerletGaussian(q_dist, p_dist, t=0.0)
        elif args.source == 'gmm':
            q_dist = GMM(nmode=args.source_nmode, device=self.device)
            p_dist = Gaussian(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))
            return VerletGMM(q_dist, p_dist, t=0.0)
        else:
            raise NotImplementedError

    def build_target(self, args):
        if args.target == 'gaussian':
            q_dist = Gaussian(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))
            p_dist = Gaussian(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))
            return VerletGaussian(q_dist, p_dist, t=1.0)
        elif args.target == 'gmm':
            q_dist = GMM(nmode=args.target_nmode, device=self.device)
            p_dist = Gaussian(torch.zeros(2, device=self.device), torch.eye(2, device=self.device))
            return VerletGMM(q_dist, p_dist, t=1.0)
        else:
            raise NotImplementedError

    def setup(self, stage):
        self.t_span = torch.linspace(0.0, 1.0, self.args.num_timesteps)

        # Initialize model
        verlet_flow = VerletFlow(data_dim=2,
                                 num_vp_hidden=self.args.num_vp_hidden,
                                 num_nvp_hidden=self.args.num_nvp_hidden,
                                 num_vp_layers=self.args.num_vp_layers,
                                 num_nvp_layers=self.args.num_nvp_layers)
        self.flow = TorchdynVerletFlow(verlet_flow)
        self.model = NeuralODE(CNFWrapper(self.flow), sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4)
        
        # Initialize source, target, train
        self.source = self.build_source(self.args)
        self.target = self.build_target(self.args)
        self.source_set = VerletDataset(self.source, 1000)
        self.target_set = VerletDataset(self.target, 1000)
        self.trainloader = data.DataLoader(self.source_set, batch_size=250, shuffle=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x0 = batch
        t_eval, trajectory = self.model(x0, self.t_span)
        x1 = trajectory[-1] # select last point of solution trajectory
        pushforward_logp = x1[:,0]
        pushforward_x = x1[:,2:]
        pushforward_data = VerletData.from_qp(pushforward_x, 1.0)
        target_logp = self.target.get_density(pushforward_data)
        loss = torch.mean(pushforward_logp - target_logp)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return self.trainloader

    # Utility functions
    def graph(self, trajectory):
        # Plot evolution
        N = len(self.t_span)
        fig, axs = plt.subplots(1, N, figsize=(8 * N,8))
        for i in range(N):
            axs[i].hist2d(trajectory[i,:,2], trajectory[i,:,3], bins=100, density=True)
        plt.show()

    def graph_marginals(self, N):
        X_test = self.source_set.sample(N)
        t_span = self.t_span
        t_eval, trajectory = self.model(X_test.cpu(), t_span.cpu())
        trajectory = trajectory.detach().cpu().numpy()
        self.graph(trajectory)

    def graph_backwards_marginals(self, N):
        X_target = self.target_set.sample(N, t=1.0)
        t_span = torch.flip(self.t_span, dims=(0,))
        t_eval, trajectory = self.model(X_target.cpu(), t_span.cpu())
        trajectory = trajectory.detach().cpu().numpy()
        self.graph(trajectory)

    @staticmethod
    def load_saved(self, path):
        return CNF.load_from_checkpoint(path)
