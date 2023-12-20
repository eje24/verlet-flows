import torch
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt

# Based on https://github.com/qsh-zh/pis/blob/master/src/datamodules/datasets/ps.py#L90
class GMM:
    def __init__(self, device, nmode=3, xlim = 3.0, scale = 0.15):
        mix = D.Categorical(torch.ones(nmode).to(device))
        angles = np.linspace(0, 2 * np.pi, nmode)
        np_loc = xlim * np.stack([np.cos(angles), np.sin(angles)]).T
        loc = torch.tensor(np_loc, dtype = torch.float).to(device)
        dist = D.Normal(loc, torch.ones(size=(nmode, 2).to(device)) * scale * xlim)
        comp = D.Independent(dist, 1)
        self.gmm = D.MixtureSameFamily(mix, comp)

    def get_density(self, x):
        return self.gmm.log_prob(x)

    def sample(self, n):
        return self.gmm.sample((n,))

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
