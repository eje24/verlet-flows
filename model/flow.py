import torch.nn as nn
import torch
from utils.distributions import log_uniform_density_so3, log_gaussian_density_r3
from scipy.stats import special_ortho_group
from coupling_layer import SE3CouplingLayer
from utils.geometry_utils import apply_update

from torch_geometric.data import Batch


def random_v_rot(batch_size):
    return torch.rand(batch_size, 3)


def random_v_tr(batch_size):
    return torch.rand(batch_size, 3)


class SE3VerletFlow(nn.Module):
    def __init__(self, device, num_layers=5):
        self.device = device
        self.num_layers = num_layers
        self.coupling_layers = nn.ModuleList(
            [SE3CouplingLayer(device) for _ in range(num_layers)])

    def log_x_latent_density(self, noise_tr: torch.Tensor):
        """
        Density of initial choice of x, as parameterized by noise_rot, noise_tr
        @param noise_rot, torch.Tensor, size: 3x3, uniformly random rotation applied to input ligand
        @param noise_tr, torch.Tensor, size: 3, Guassian translation applied to input ligand from protein center of mass
        @returns log density
        """
        return log_uniform_density_so3() + log_gaussian_density_r3(noise_tr)

    def log_v_latent_density(self, v_rot: torch.Tensor, v_tr: torch.Tensor):
        """
        Density of initial choice of v, as parameterized by v_rot, v_tr
        @param v_rot, torch.Tensor, size: 3, axis-angle representation of rotation
        @param v_tr, torch.Tensor, size: 3, translation
        @returns log density
        """
        return log_gaussian_density_r3(v_rot) + log_gaussian_density_r3(v_tr)

    def _forward(self, x: Batch, v_rot: torch.Tensor, v_tr: torch.Tensor, log_pxv_initial: torch.Tensor):
        """
        @param x, torch_geometric.data.Batch, batch of protein-ligand pairs, with noise added
        @param v_rot, torch.Tensor, batch_size x self.v_rot_dim, rotation vectors
        @param v_tr, torch.Tensor, batch_size x self.v_tr_dim, translation vectors
        @param log_pxv_initial, torch.Tensor, log densities of sampled x,v_rot,v_tr structures
        @returns x, sampled structure if reverse=False, otherwise latent structure
        @returns v_rot, sampled v_rot if reverse=False, otherwise latent v_rot
        @returns v_tr, sampled v_tr if reverse=False, otherwise latent v_tr
        @returns delta_log_pxv, torch.Tensor, change in log p(x,v) from input to output
        """
        delta_log_pxv = 0
        for coupling_layer in self.coupling_layers:
            x, v_rot, v_tr, _delta_log_pxv = coupling_layer(
                x, v_rot, v_tr, reverse=False)
            delta_log_pxv += _delta_log_pxv
        return x, v_rot, v_tr, log_pxv_initial + delta_log_pxv

    def forward(self, x: Batch, v_rot: torch.Tensor, v_tr: torch.Tensor, noise_rot: torch.Tensor, noise_tr: torch.Tensor):
        """
        Wrapper around _forward
        @param x, torch_geometric.data.Batch, batch of protein-ligand pairs, with no noise added
        @returns x_sampled, torch_geometric.data.Batch, sampled batch of protein-ligand pairs
        @returns log_pxv = log likelihood of x wrt to choice of v if reverse = True, 
            log likelihood
        """
        log_pxv = torch.zeros(x.num_graphs)
        for i, complex in enumerate(x.to_data_list()):
            # sample translation from Gaussian centered at center of mass of protein
            protein_center = torch.mean(
                complex['protein'].pos, axis=0, keepdims=True)
            ligand_center = torch.mean(
                complex['ligand'].pos, axis=0, keepdims=True)
            complex = apply_update(
                complex, noise_rot[i], protein_center - ligand_center + noise_tr)
            log_pxv[i] = self.log_x_latent_density(
                noise_tr[i]) + self.log_v_latent_density(v_rot[i], v_tr[i])
        return self._forward(
            x, v_rot, v_tr, log_pxv)

    def _reverse(self, x: Batch, v_rot: torch.Tensor, v_tr: torch.Tensor):
        log_pxv = torch.zeros(x.num_graphs)
        for coupling_layer in self.coupling_layers:
            x, v_rot, v_tr, delta_log_pxv = coupling_layer(
                x, v_rot, v_tr, reverse=True)
            log_pxv += delta_log_pxv
        # determine initial probabilities
        for i, complex in enumerate(x.to_data_list()):
            protein_center = torch.mean(
                complex['protein'].pos, axis=0, keepdims=True)
            ligand_center = torch.mean(
                complex['ligand'].pos, axis=0, keepdims=True)
            log_pxv[i] += self.log_x_latent_density(
                ligand_center - protein_center)
            log_pxv[i] += self.log_v_latent_density(v_rot[i], v_tr[i])
        return x, v_rot, v_tr, log_pxv

    def reverse(self, x: Batch, v_rot: torch.Tensor, v_tr: torch.Tensor):
        """
        Samples a random velocity vector, computed inverse under flow, and returns probability
        """
        return self._reverse(x, v_rot, v_tr)
