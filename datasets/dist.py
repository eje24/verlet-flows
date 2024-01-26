from abc import ABC, abstractmethod

import torch
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from datasets.aug_data import AugmentedData

def graph_log_density(density_fn, ax, bins=100, xlim=5, ylim=5):
    x = np.linspace(-xlim, xlim, bins)
    y = np.linspace(-ylim, ylim, bins)
    X, Y = np.meshgrid(x, y)
    xy = np.stack([X.reshape(-1), Y.reshape(-1)]).T
    density = density_fn(torch.tensor(xy, device='cuda')).cpu().numpy().reshape(bins, bins)
    ax.imshow(density, extent=[-xlim, xlim, -ylim, ylim])

def graph_log_grad_fn(log_grad_fn, ax, bins=25, xlim=5, ylim=5, scale=30.0):
    x = np.linspace(-xlim, xlim, bins)
    y = np.linspace(-ylim, ylim, bins)
    X, Y = np.meshgrid(x, y)
    xy = np.stack([X.reshape(-1), Y.reshape(-1)]).T
    graph = log_grad_fn(torch.tensor(xy, device='cuda')).cpu().numpy().reshape(bins, bins, 2)
    ax.quiver(X, Y, graph[:, :, 0], graph[:, :, 1], scale=30.0)


class Density(ABC):
    @abstractmethod
    def get_log_density(self, x):
        pass

    @abstractmethod
    def log_grad_fn(self, x):
        pass

class Sampleable(ABC):
    @abstractmethod
    def sample(self, n):
        pass

class Distribution(Density, Sampleable):
    pass



class Gaussian(Distribution):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.gaussian = D.MultivariateNormal(mean, cov)
        self.dim = mean.shape[0]

    def to(self, device):
        self.mean = self.mean.to(device)
        self.cov = self.cov.to(device)
        self.gaussian = D.MultivariateNormal(self.mean, self.cov)
        return self

    def sample(self, n):
        return self.gaussian.sample((n,))

    def get_log_density(self, x):
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

    def graph_log_density_and_grad(self):
        ax = plt.subplot(1, 1, 1)
        graph_log_density(self.get_log_density, ax)
        graph_log_grad_fn(self.log_grad_fn, ax)

    # Gradient of the density
    def grad_fn(self, x: torch.Tensor):
        return self.log_grad_fn(x) * torch.exp(self.get_log_density(x)).view(-1, 1)


    # Gradient of the log density
    def log_grad_fn(self, x: torch.Tensor):
        return -torch.matmul((x.float() - self.mean), torch.inverse(self.cov))

class GMM(Distribution):
    def __init__(self, means: list[torch.Tensor], covs: list[torch.Tensor], weights: list[float], device = None):
        self.nmodes = len(means)
        self.gaussians = [Gaussian(mean.float(), cov.float()) for mean, cov in zip(means, covs)]
        self.weights = weights
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(device)

    @staticmethod
    def regular_gmm(nmode, xlim = 3.0, scale = 0.15):
        angles = np.linspace(0, 2 * np.pi, nmode+1)[:-1]
        means = xlim * np.stack([np.cos(angles), np.sin(angles)]).T
        means = [torch.from_numpy(means[idx]) for idx in range(nmode)]
        covs = [torch.eye(2) * scale * xlim for _ in range(nmode)]
        weights = [1.0 / nmode for _ in range(nmode)]
        return GMM(means, covs, weights)

    def to(self, device):
        self.device = device
        # Move gaussians to device
        for idx in range(self.nmodes):
            self.gaussians[idx] = self.gaussians[idx].to(device)

    def sample(self, n):
        # Sample from categorical distribution given by weights
        sample_idxs = torch.multinomial(torch.tensor(self.weights), n, replacement=True).to(self.device)
        samples = torch.zeros(n,2).to(self.device)
        for idx in range(self.nmodes):
            # Get samples from each gaussian
            gaussian_samples = self.gaussians[idx].sample(n)
            # Select samples from this gaussian
            gaussian_idxs = (sample_idxs == idx)
            # Replace samples
            samples[gaussian_idxs] = gaussian_samples[gaussian_idxs]
        return samples

    # Returns log density
    def get_log_density(self, x):
        x = x.to(self.device)
        density = torch.zeros(x.shape[0]).to(self.device)
        for idx, gaussian in enumerate(self.gaussians):
            density = density + self.weights[idx] * torch.exp(gaussian.get_log_density(x))
        return torch.log(density)

    def graph_log_density(self, bins=100):
        # Use np.meshgrid to create a grid of points
        x = np.linspace(-3, 3, bins)
        y = np.linspace(-3, 3, bins)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.reshape(-1), Y.reshape(-1)]).T
        density = self.get_log_density(torch.tensor(xy, device=self.device)).cpu().numpy().reshape(bins, bins)
        plt.imshow(density, extent=[-5, 5, -5, 5])

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

    @torch.no_grad()
    def graph_log_density_and_grad(self):
        ax = plt.subplot(1, 1, 1)
        graph_log_density(self.get_log_density, ax)
        graph_log_grad_fn(self.log_grad_fn, ax, bins=20, scale=250.0)

    def log_grad_fn(self, x: torch.Tensor):
        x = x.to(self.device)
        grad = torch.zeros_like(x).to(x)
        norm = torch.zeros(x.shape[0]).to(x)
        for idx, gaussian in enumerate(self.gaussians):
            grad = grad + self.weights[idx] * gaussian.grad_fn(x)
            norm = norm + self.weights[idx] * torch.exp(gaussian.get_log_density(x))
        return grad / norm.view(-1,1)

class Funnel(Distribution):
    def __init__(self, device, dim=10):
        assert dim > 1
        self.dim = dim
        self.device = device

    def to(self, device):
        self.device = device

    def sample(self, n):
        # Sample x0 from N(0, self.dim - 1)
        x1 = torch.randn(n, 1, device=self.device)
        # Sample x1:xdim from N(0, exp(x0)I)
        x2_D = torch.randn(n, self.dim - 1, device=self.device) * torch.exp(x1 / 2)
        return torch.cat([x1, x2_D], dim=1)

    def get_log_density(self, x):
        # Split x
        x1 = x[:, 0]
        x2_D = x[:, 1:]
        # Get densities for each part
        x1_density = D.Normal(0, 1).log_prob(x1)
        # Get densities for remaining parts
        x1_var = torch.exp(x1) # (B, )
        x2_D_mean = torch.zeros(self.dim - 1, device=self.device) # (D - 1, )
        x2_D_covar = x1_var.view(-1, 1, 1) * torch.eye(self.dim - 1, device=self.device).view(1, self.dim - 1, self.dim - 1) # (B, D - 1, D - 1)
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

    def graph_log_density(self, bins=1000):
        assert self.dim == 2, "Can only graph 2D funnels"
        # Use np.meshgrid to create a grid of points
        x = np.linspace(-5, 5, bins)
        y = np.linspace(-10, 10, bins)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.reshape(-1), Y.reshape(-1)]).T
        density = self.get_log_density(torch.tensor(xy, device=self.device)).cpu().numpy().reshape(bins, bins)
        # Clip density for better visualization
        density = np.clip(density, -10, 0)
        plt.imshow(density, extent=[-5, 5, -10, 10])
        plt.colorbar()

    # Gradient of the log density
    def log_grad_fn(self, x: torch.Tensor):
        raise NotImplementedError

class AugmentedDistribution(Distribution):
    def __init__(self, q_dist: Density, p_dist: Density, t: float = 1.0):
        self.q_dist = q_dist
        self.p_dist = p_dist
        self.t = t

    def to(self, device):
        self.q_dist.to(device)
        self.p_dist.to(device)
        return self

    def sample(self, n):
        q = self.q_dist.sample(n)
        p = self.p_dist.sample(n)
        t = self.t * torch.ones((n,1), device=q.device)
        return AugmentedData(q, p, t)

    def get_log_density(self, data: AugmentedData):
        q = self.q_dist.get_log_density(data.q)
        p = self.p_dist.get_log_density(data.p)
        return q + p

    def log_grad_fn(self, data: AugmentedData):
        q_grad = self.q_dist.log_grad_fn(data.q)
        p_grad = self.p_dist.log_grad_fn(data.p)
        return torch.cat([q_grad, p_grad], dim=1)

def build_augmented_distribution(cfg: DictConfig, device, t: float = 1.0) -> AugmentedDistribution:
    p_dist = Gaussian(torch.zeros(cfg.dim), torch.eye(cfg.dim))
    if cfg.distribution == 'gaussian':
        q_dist = Gaussian(torch.zeros(cfg.dim), torch.eye(cfg.dim))
    elif cfg.distribution == 'gmm':
        q_dist = GMM.regular_gmm(nmode=cfg.nmode)
    elif cfg.distribution == 'funnel':
        q_dist = Funnel(device, dim=cfg.dim)
    elif cfg.distribution == 'weird_gaussian':
        q_dist = Gaussian(2.0 * torch.ones(2, device=device), torch.tensor([[5.0, 2.0], [2.0, 1.0]], device=device))
    elif cfg.distribution == 'weird_gmm':
        raise NotImplementedError
    elif cfg.distribution == 'lgcp':
        raise NotImplementedError
    return AugmentedDistribution(q_dist, p_dist, t=t)


