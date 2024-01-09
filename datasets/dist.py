from abc import ABC, abstractmethod
import math

import torch
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt

from datasets.verlet import VerletData

class Density(ABC):
    @abstractmethod
    def get_density(self, x):
        pass

class Sampleable(ABC):
    @abstractmethod
    def sample(self, n):
        pass


# Based on https://github.com/qsh-zh/pis/blob/master/src/datamodules/datasets/ps.py#L90
class GMM(Density, Sampleable):
    def __init__(self, device, nmode=3, xlim = 3.0, scale = 0.15):
        mix = D.Categorical(torch.ones(nmode).to(device))
        angles = np.linspace(0, 2 * np.pi, nmode+1)[:-1]
        np_loc = xlim * np.stack([np.cos(angles), np.sin(angles)]).T
        loc = torch.tensor(np_loc, dtype = torch.float).to(device)
        dist = D.Normal(loc, torch.ones(size=(nmode, 2), device=device) * scale * xlim)
        comp = D.Independent(dist, 1)
        self.device = device
        self.gmm = D.MixtureSameFamily(mix, comp)

    def get_density(self, x):
        return self.gmm.log_prob(x)

    def sample(self, n):
        return self.gmm.sample((n,))

    def graph_density(self, bins=100):
        # Use np.meshgrid to create a grid of points
        x = np.linspace(-3, 3, bins)
        y = np.linspace(-3, 3, bins)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.reshape(-1), Y.reshape(-1)]).T
        density = self.get_density(torch.tensor(xy, device=self.device)).cpu().numpy().reshape(bins, bins)
        plt.imshow(density, extent=[-3, 3, -3, 3])

    def graph(self, N, bins=100):
        # Sample N points
        samples = self.sample(N).cpu().numpy()
        
        # Use the samples to create a heatmap
        plt.hist2d(samples[:, 0], samples[:, 1], bins=bins, density=True)
        
        # Add a color bar to the side
        plt.colorbar()

        # Set the aspect of the plot to be equal
        plt.gca().set_aspect('equal', adjustable='box')

        # Show the plot
        plt.show()


class Gaussian(Sampleable):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.gaussian = D.MultivariateNormal(mean, cov)
        self.dim = mean.shape[0]

    def sample(self, n):
        return self.gaussian.sample((n,))

    def get_density(self, x):
        return self.gaussian.log_prob(x)

    @staticmethod
    def random(device) -> 'Gaussian':
        mean = torch.rand(2).to(device)
        pre_cov = torch.rand(2, 2).to(device)
        cov = torch.mm(pre_cov, pre_cov.t())
        return Gaussian(mean, cov)

    def graph(self, N, bins=100):
        # Sample N points
        samples = self.sample(N).cpu().numpy()
        
        # Use the samples to create a heatmap
        plt.hist2d(samples[:, 0], samples[:, 1], bins=bins, density=True)
        
        # Add a color bar to the side
        plt.colorbar()

        # Set the aspect of the plot to be equal
        plt.gca().set_aspect('equal', adjustable='box')

        # Show the plot
        plt.show()

class Funnel(Sampleable, Density):
    def __init__(self, device, dim=10):
        assert dim > 1
        self.dim = dim
        self.device = device

    def sample(self, n):
        # Sample x0 from N(0, self.dim - 1)
        x1 = torch.randn(n, 1, device=self.device)
        # Sample x1:xdim from N(0, exp(x0)I)
        x2_D = torch.randn(n, self.dim - 1, device=self.device) * torch.exp(x1 / 2)
        return torch.cat([x1, x2_D], dim=1)

    def get_density(self, x):
        # Split x
        x1 = x[:, 0]
        x2_D = x[:, 1:]
        # Get densities for each part
        x1_density = D.Normal(0, 1).log_prob(x1)
        # Get densities for remaining parts
        x1_var = torch.exp(x1)
        x2_D_mean = torch.zeros(self.dim - 1, device=self.device)
        x2_D_covar = x1_var.view(-1, 1, 1) * torch.eye(self.dim - 1, device=self.device).view(1, self.dim - 1, self.dim - 1)
        x2_D_density = D.MultivariateNormal(x2_D_mean, x2_D_covar).log_prob(x2_D)
        return x1_density + x2_D_density

    def graph(self, N, bins=100):
        assert self.dim == 2, "Can only graph 2D funnels"
        # Sample N points
        samples = self.sample(N).cpu().numpy()
        
        # Use the samples to create a heatmap
        plt.hist2d(samples[:, 0], samples[:, 1], bins=bins, density=True)
        
        # Add a color bar to the side
        plt.colorbar()

        # Set the aspect of the plot to be equal
        plt.gca().set_aspect('equal', adjustable='box')

        # Show the plot
        plt.show()

    def graph_density(self, bins=1000):
        assert self.dim == 2, "Can only graph 2D funnels"
        # Use np.meshgrid to create a grid of points
        x = np.linspace(-5, 5, bins)
        y = np.linspace(-10, 10, bins)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.reshape(-1), Y.reshape(-1)]).T
        density = self.get_density(torch.tensor(xy, device=self.device)).cpu().numpy().reshape(bins, bins)
        # Clip density for better visualization
        density = np.clip(density, -10, 0)
        plt.imshow(density, extent=[-5, 5, -10, 10])
        plt.colorbar()


class VerletGMM(Density):
    def __init__(self, q_dist: GMM, p_dist: Gaussian, t: float = 1.0):
        self.q_dist = q_dist
        self.p_dist = p_dist
        self.t = t

    def sample(self, n):
        q = self.q_dist.sample(n)
        p = self.p_dist.sample(n)
        t = self.t * torch.ones((n,1), device=q.device)
        return VerletData(q, p, t)

    def get_density(self, data: VerletData):
        q = self.q_dist.get_density(data.q)
        p = self.p_dist.get_density(data.p)
        return q + p

class VerletGaussian(Sampleable, Density):
    def __init__(self, q_dist: Gaussian, p_dist: Gaussian, t: float = 1.0):
        self.q_dist = q_dist
        self.p_dist = p_dist
        self.t = t

    def sample(self, n):
        q = self.q_dist.sample(n)
        p = self.p_dist.sample(n)
        t = self.t * torch.ones((n,1), device=q.device)
        return VerletData(q, p, t)

    def get_density(self, data: VerletData):
        q = self.q_dist.get_density(data.q)
        p = self.p_dist.get_density(data.p)
        return q + p

class VerletFunnel(Sampleable, Density):
    def __init__(self, q_dist: Funnel, p_dist: Gaussian, t: float = 1.0):
        self.q_dist = q_dist
        self.p_dist = p_dist
        self.t = t

    def sample(self, n):
        q = self.q_dist.sample(n)
        p = self.p_dist.sample(n)
        t = self.t * torch.ones((n,1), device=q.device)
        return VerletData(q, p, t)

    def get_density(self, data: VerletData):
        q = self.q_dist.get_density(data.q)
        p = self.p_dist.get_density(data.p)
        return q + p



