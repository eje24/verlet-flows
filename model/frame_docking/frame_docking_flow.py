import torch.nn as nn
import torch

from utils.distributions import log_uniform_density_so3, log_gaussian_density_r3
from utils.geometry_utils import axis_angle_to_matrix, apply_update
from model.frame_docking.frame_score_model import FrameDockingScoreModel
from datasets.frame_dataset import VerletFrame


class FrameDockingVerletFlow(nn.Module):
    """
    SE3 Verlet flow for frame docking
    """

    def __init__(self, num_coupling_layers=5, **kwargs):
        super().__init__()
        self.num_coupling_layers = num_coupling_layers
        self.coupling_layers = nn.ModuleList(
            [FrameDockingCouplingLayer(**kwargs) for _ in range(num_coupling_layers)]
        )

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
            data, _delta_log_pxv = coupling_layer(data, reverse=False)
            delta_log_pxv = delta_log_pxv + _delta_log_pxv
        return data, log_pxv_initial + delta_log_pxv, delta_log_pxv

    def latent_to_data(
        self, data: VerletFrame, noise_rot: torch.Tensor, noise_tr: torch.Tensor
    ):
        """
        Wrapper around _latent_to_data,
        computes image of data under the flow and log densities

        Args:
            data: ligand/protein structure
            noise_rot: noising rotation to apply to ligand


        Returns:
            sampled poses, and
            log densities
            delta log densities
        """
        log_pxv = torch.zeros(data.num_graphs, device=next(self.parameters()).device)
        protein_center = torch.mean(data.receptor, axis=1)
        ligand_center = torch.mean(data.ligand, axis=1)
        apply_update(data, noise_rot, protein_center - ligand_center + noise_tr)
        log_pxv = self.log_x_latent_density(noise_tr) + self.log_v_latent_density(
            data.v_rot, data.v_tr
        )
        return self._latent_to_data(data, log_pxv)

    def _data_to_latent(self, data: VerletFrame):
        """
        Computes the pre-image of data under the flow and the latent log densities

        Args:
            data: ligand/protein structures

        Returns:
            latent poses of ligand/protein structures passed in, and
            log densities of latent poses
        """
        log_pxv = torch.zeros(data.ligand.shape[0], device=data.ligand.device)
        for coupling_layer in reversed(self.coupling_layers):
            data, delta_log_pxv = coupling_layer(data, reverse=True)
            log_pxv = log_pxv + delta_log_pxv
        delta_log_pxv = log_pxv.detach().clone()

        protein_center = torch.mean(data.receptor, dim=1)
        ligand_center = torch.mean(data.ligand, dim=1)
        log_pxv = log_pxv + self.log_x_latent_density(
            (ligand_center - protein_center).reshape((-1, 1, 3))
        )
        log_pxv = log_pxv + self.log_v_latent_density((data.v_rot, data.v_tr))
        return data, log_pxv, delta_log_pxv

    def data_to_latent(self, data: VerletFrame):
        """
        Computes the pre-image of data under the flow and the latent log densities

        Args:
            data: ligand/protein structures

        Returns:
            latent poses of ligand/protein structures passed in, and
            log densities of latent poses
        """
        return self._data_to_latent(data)

    def forward(self, data):
        _, log_pxv, _ = self.data_to_latent(data)
        return log_pxv

    def check_invertible(self, data: VerletFrame):
        """
        Checks that flow is invertible

        Args:
            data: ligand/protein structures
        """
        data_pos = data.detach().clone()
        data_v_rot = data.v_tr.detach().clone()
        data_v_tr = data.v_rot.detach().clone()

        print(f"Initial v_rot: {data_v_rot}")
        print(f"Initial v_tr: {data_v_tr}")

        self.eval()
        latent_data, log_pxv, reverse_delta_log_pxv = self._data_to_latent(data)
        recreated_data, recreated_log_pxv, forward_delta_log_pxv = self._latent_to_data(
            latent_data, log_pxv - reverse_delta_log_pxv
        )
        self.train()

        recreated_data_pos = recreated_data.detach().clone()
        recreated_data_v_rot = recreated_data.v_tr.detach().clone()
        recreated_data_v_tr = recreated_data.v_rot.detach().clone()

        print(f"Recreated v_rot: {recreated_data_v_rot}")
        print(f"Recreated v_tr: {recreated_data_v_tr}")

        assert torch.allclose(data_pos, recreated_data_pos, rtol=1e-05, atol=1e-05)
        assert torch.allclose(data_v_rot, recreated_data_v_rot, rtol=1e-05, atol=1e-05)
        assert torch.allclose(data_v_tr, recreated_data_v_tr, rtol=1e-05, atol=1e-05)
        assert torch.allclose(
            reverse_delta_log_pxv, forward_delta_log_pxv, rtol=1e-05, atol=1e-05
        )
        assert torch.allclose(log_pxv, recreated_log_pxv, rtol=1e-05, atol=1e-05)


class FrameDockingCouplingLayer(nn.Module):
    """
    SE3 Verlet coupling layer for docking
    """

    def __init__(self, device, **kwargs):
        super().__init__()
        self.st_net = FrameDockingScoreModel(**kwargs)
        # constrain timestep to be >0
        self.log_timestep = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, data: VerletFrame, reverse: bool = False):
        """
        Implements a single SE3 Verlet coupling layer

        Args:
            data: ligand/protein structures

        Returns:
            change in log densities
        """
        # compute timestep
        timestep = torch.exp(self.log_timestep)
        if not reverse:
            # forward coupling step
            s_rot, s_tr, t_rot, t_tr = self.st_net(data)
            data.v_rot = data.v_rot * torch.exp(s_rot) + t_rot
            data.v_tr = data.v_tr * torch.exp(s_tr) + t_tr
            print(
                f"Devices: timestep is {timestep.device}, data.v_rot is {data.v_rot.device}"
            )
            update_rot = timestep * data.v_rot
            update_tr = timestep * data.v_tr
            v_rot_matrix = axis_angle_to_matrix(update_rot).squeeze()
            apply_update(data, v_rot_matrix, update_tr)
            delta_pxv = -torch.sum(s_rot, axis=-1) - torch.sum(s_tr, axis=-1)
            return data, delta_pxv
        else:
            # backward coupling step
            print(
                f"Devices: timestep is {timestep.device}, data.v_rot is {data.v_rot.device}"
            )
            print(f"Shape of data.ligand: {data.ligand.shape}")
            update_rot = timestep * data.v_rot
            update_tr = timestep * data.v_tr
            negative_v_rot_matrix = axis_angle_to_matrix(-update_rot).squeeze()
            apply_update(data, negative_v_rot_matrix, -data.v_tr)
            print(f"Shape of data.ligand: {data.ligand.shape}")
            # all shapes batch_size x 3
            s_rot, s_tr, t_rot, t_tr = self.st_net(data)
            breakpoint()
            data.v_rot = (data.v_rot - t_rot) * torch.exp(-s_rot)
            data.v_tr = (data.v_tr - t_tr) * torch.exp(-s_tr)
            delta_pxv = -torch.sum(s_rot, axis=-1) - torch.sum(s_tr, axis=-1)
            return data, delta_pxv

    def check_invertibility(self, data: VerletFrame):
        """
        Checks that coupling layer is invertible

        Args:
            data: ligand/protein structures
        """
        data_pos = data.detach().clone()
        data_v_rot = data.v_tr.detach().clone()
        data_v_tr = data.v_rot.detach().clone()

        print(f"Initial v_rot: {data_v_rot}")
        print(f"Initial v_tr: {data_v_tr}")

        self.eval()
        # go forward
        data, _ = self.forward(data, reverse=False)
        # go backwards
        recreated_data, _ = self.forward(data, reverse=True)
        self.train()

        recreated_data_pos = recreated_data.detach().clone()
        recreated_data_v_rot = recreated_data.v_tr.detach().clone()
        recreated_data_v_tr = recreated_data.v_rot.detach().clone()

        print(f"Recreated v_rot: {recreated_data_v_rot}")
        print(f"Recreated v_tr: {recreated_data_v_tr}")

        assert torch.allclose(data_pos, recreated_data_pos, rtol=1e-05, atol=1e-05)
        assert torch.allclose(data_v_rot, recreated_data_v_rot, rtol=1e-05, atol=1e-05)
        assert torch.allclose(data_v_tr, recreated_data_v_tr, rtol=1e-05, atol=1e-05)
