from abc import ABC, abstractmethod

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

class VerletGMM(Density):
    def __init__(self, q_density: GMM, p_density: Gaussian):
        self.q_density = q_density
        self.p_density = p_density

    def get_density(self, data: VerletData):
        q = self.q_density.get_density(data.q)
        p = self.p_density.get_density(data.p)
        return q + p

class VerletGaussian(Sampleable, Density):
    def __init__(self, q_sampleable: Gaussian, p_sampleable: Gaussian):
        self.q_sampleable = q_sampleable
        self.p_sampleable = p_sampleable

    def sample(self, n):
        q = self.q_sampleable.sample(n)
        p = self.p_sampleable.sample(n)
        return VerletData(q, p, 0.0)

    def get_density(self, data: VerletData):
        q = self.q_sampleable.get_density(data.q)
        p = self.p_sampleable.get_density(data.p)
        return q + p


