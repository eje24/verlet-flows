import torch.nn as nn
import torch
from utils.distributions import log_uniform_density_so3, log_gaussian_density_r3
from model.coupling_layer import SE3CouplingLayer
from utils.geometry_utils import apply_update

from torch_geometric.data import Batch


class SE3VerletFlow(nn.Module):
    def __init__(self, device, num_coupling_layers=5, **kwargs):
        super().__init__()
        self.device = device
        self.num_coupling_layers = num_coupling_layers
        self.coupling_layers = nn.ModuleList(
            [SE3CouplingLayer(device=device, **kwargs) for _ in range(num_coupling_layers)])

    def log_x_latent_density(self, noise_tr: torch.Tensor):
        """
        Density of initial choice of x, as parameterized by noise_rot, noise_tr
        
        Args:
            noise_rot: size: 3x3, uniformly random rotation applied to input ligand
            noise_tr: size: 3, Guassian translation applied to input ligand from protein center of mass
        Returns:
            log density
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
        Computes image of data under the flow and log densities
        
        Args:
            data: ligand/protein structure
            noise_rot: noising rotation to apply to ligand
            noise_tr: noising translation to apply to ligand
            
        Returns:
            sampled poses,
            log densities, and
            delta log densities
        """
        delta_log_pxv = 0
        for coupling_layer in self.coupling_layers:
            data, _delta_log_pxv = coupling_layer(
                data, reverse=False)
            delta_log_pxv = delta_log_pxv + _delta_log_pxv
        return data, log_pxv_initial + delta_log_pxv, delta_log_pxv

    def latent_to_data(self, data: Batch, noise_rot: torch.Tensor, noise_tr: torch.Tensor):
        """
        Wrapper around _latent_to_data, 
        computes image of data under the flow and log densities
        
        Args:
            data: ligand/protein structure
            noise_rot: noising rotation to apply to ligand
            noise_tr: noising translation to apply to ligand
            
        Returns:
            sampled poses, and
            log densities
            delta log densities
        """
        log_pxv = torch.zeros(data.num_graphs, device = next(self.parameters()).device)
        for i, complex in enumerate(data.to_data_list()):
            # sample translation from Gaussian centered at center of mass of protein
            protein_center = torch.mean(complex['receptor'].pos, axis=0, keepdims=True)
            ligand_center = torch.mean(complex['ligand'].pos, axis=0, keepdims=True)
            complex = apply_update(complex, noise_rot[i], protein_center - ligand_center + noise_tr)
            log_pxv[i] = self.log_x_latent_density(noise_tr[i]) + self.log_v_latent_density(data.v_rot[i], data.v_tr[i])
        return self._latent_to_data(data, log_pxv)

    def _data_to_latent(self, data: Batch):
        """
        Computes the pre-image of data under the flow and the latent log densities
        
        Args:
            data: ligand/protein structures
        
        Returns:
            latent poses of ligand/protein structures passed in, and
            log densities of latent poses
        """
        log_pxv = torch.zeros(data.num_graphs, device = data['ligand'].pos.device)
        for coupling_layer in reversed(self.coupling_layers):
            data, delta_log_pxv = coupling_layer(
                data, reverse=True)
            log_pxv = log_pxv + delta_log_pxv
        delta_log_pxv = log_pxv.detach().clone()
        # determine initial densities
        for i, complex in enumerate(data.to_data_list()):
            protein_center = torch.mean(complex['receptor'].pos, axis=0, keepdims=True)
            ligand_center = torch.mean(complex['ligand'].pos, axis=0, keepdims=True)
            log_pxv[i] = log_pxv[i] + self.log_x_latent_density((ligand_center - protein_center).reshape((1,3)))
            log_pxv[i] = log_pxv[i] + self.log_v_latent_density(data.v_rot[i].reshape((1,3)), data.v_tr[i].reshape((1,3)))
        return data, log_pxv, delta_log_pxv

    def data_to_latent(self, data: Batch):
        """
        Computes the pre-image of data under the flow and the latent log densities
        
        Args:
            data: ligand/protein structures
        
        Returns:
            latent poses of ligand/protein structures passed in, and
            log densities of latent poses
        """
        return self._data_to_latent(data)
    
    def forward(self, data: Batch):
        _, log_pxv, _ = self.data_to_latent(data)
        return log_pxv
    
    def check_invertible(self, data: Batch): 
        """
        Checks that flow is invertible
        
        Args:
            data: ligand/protein structures
        """
        data_pos = data['ligand'].pos.detach().clone()
        data_v_rot = data.v_tr.detach().clone()
        data_v_tr = data.v_rot.detach().clone()
        
        print(f'Initial v_rot: {data_v_rot}')
        print(f'Initial v_tr: {data_v_tr}')
        
        self.eval()
        latent_data, log_pxv, reverse_delta_log_pxv =  self._data_to_latent(data)
        recreated_data, recreated_log_pxv, forward_delta_log_pxv = self._latent_to_data(latent_data, log_pxv - reverse_delta_log_pxv)
        self.train()
        
        recreated_data_pos = recreated_data['ligand'].pos.detach().clone()
        recreated_data_v_rot = recreated_data.v_tr.detach().clone()
        recreated_data_v_tr = recreated_data.v_rot.detach().clone()   
        
        print(f'Recreated v_rot: {recreated_data_v_rot}')
        print(f'Recreated v_tr: {recreated_data_v_tr}')
        
        assert torch.allclose(data_pos, recreated_data_pos, rtol=1e-05, atol=1e-05)
        assert torch.allclose(data_v_rot, recreated_data_v_rot, rtol=1e-05, atol=1e-05)
        assert torch.allclose(data_v_tr, recreated_data_v_tr, rtol=1e-05, atol=1e-05)
        assert torch.allclose(reverse_delta_log_pxv, forward_delta_log_pxv, rtol=1e-05, atol=1e-05)
        assert torch.allclose(log_pxv, recreated_log_pxv, rtol=1e-05, atol=1e-05)

        
        
        
            
        
