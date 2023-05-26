import torch.nn as nn
import torch
from utils.distributions import log_uniform_density_so3, log_gaussian_density_r3
from model.coupling_layer import SE3CouplingLayer
from utils.geometry_utils import apply_update

from torch_geometric.data import Batch


class SE3VerletFlow(nn.Module):
    def __init__(self, num_coupling_layers=5, **kwargs):
        super().__init__()
        self.num_coupling_layers = num_coupling_layers
        self.coupling_layers = nn.ModuleList(
            [SE3CouplingLayer(**kwargs) for _ in range(num_coupling_layers)])

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
        
        Args:
            v_rot, torch.Tensor, size: 3, axis-angle representation of rotation
            v_tr, torch.Tensor, size: 3, translation
        Returns:
            log density
        """
        return log_gaussian_density_r3(v_rot) + log_gaussian_density_r3(v_tr)

    def _latent_to_data(self, data, log_pxv_initial: torch.Tensor):
        """
        """
        delta_log_pxv = 0
        for coupling_layer in self.coupling_layers:
            data, _delta_log_pxv = coupling_layer(
                data, reverse=False)
            delta_log_pxv += _delta_log_pxv
        return data, log_pxv_initial + delta_log_pxv

    def latent_to_data(self, data: Batch, noise_rot: torch.Tensor, noise_tr: torch.Tensor):
        """
        Wrapper around _forward
        """
        log_pxv = torch.zeros(data.num_graphs)
        for i, complex in enumerate(data.to_data_list()):
            # sample translation from Gaussian centered at center of mass of protein
            protein_center = torch.mean(complex['protein'].pos, axis=0, keepdims=True)
            ligand_center = torch.mean(complex['ligand'].pos, axis=0, keepdims=True)
            complex = apply_update(
                complex, noise_rot[i], protein_center - ligand_center + noise_tr)
            log_pxv[i] = self.log_x_latent_density(
                noise_tr[i]) + self.log_v_latent_density(data.v_rot[i], data.v_tr[i])
        return self._latent_to_data(data, log_pxv)

    def _data_to_latent(self, data: Batch):
        """
        @param data, Batch, data
        """
        log_pxv = torch.zeros(data.num_graphs)
        for coupling_layer in self.coupling_layers:
            data, delta_log_pxv = coupling_layer(
                data, reverse=True)
            log_pxv += delta_log_pxv
        # determine initial probabilities
        for i, complex in enumerate(data.to_data_list()):
            protein_center = torch.mean(complex['protein'].pos, axis=0, keepdims=True)
            ligand_center = torch.mean(complex['ligand'].pos, axis=0, keepdims=True)
            log_pxv[i] += self.log_x_latent_density(
                ligand_center - protein_center)
            log_pxv[i] += self.log_v_latent_density(
                data.v_rot[i], data.v_tr[i])
        return data, log_pxv

    def data_to_latent(self, data: Batch):
        """
        Samples a random velocity vector, computed inverse under flow, and returns probability
        """
        return self._data_to_latent(data)
    
    def forward(self, data: Batch):
        return self.data_to_latent(data)
            
        
