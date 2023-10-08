from enum import Enum
from typing import List, Optional, Tuple

import torch.nn as nn
import torch

from utils.geometry_utils import axis_angle_to_matrix, apply_update
from model.frame_docking.frame_score_model import FrameDockingScoreModel
from datasets.frame_dataset import VerletFrame, FramePrior


class FlowTrajectoryMetadata:
    """
    trajectory: sequence of Verlet frames from prior to data distribution
    flow_logp: the change in log density as a result of the flow, (torch.Tensor, rather than float, because frames are batched)
    prior_logp: the prior log density (torch.Tensor, rather than float, because frames are batched)
    """
    def __init__(self):
        self.trajectory: List[torch.Tensor] = list()
        self.flow_logp: Optional[torch.Tensor] = None
        self.prior_logp: Optional[torch.Tensor] = None

    def total_logp(self) -> Optional[float]:
        assert self.flow_logp is not None and self.prior_logp is not None
        return self.flow_logp + self.prior_logp
    
class FlowWrapper(nn.Module):
    def __init__(self, flow: "FrameDockingVerletFlow", prior: FramePrior):
        super().__init__()
        self._flow = flow
        self._prior = prior

    def latent_to_data(self, latent: VerletFrame) -> Tuple[VerletFrame, FlowTrajectoryMetadata]:
        # get prior density
        prior_logp = self._prior.get_logp(latent)
        # run flow
        data, flow_trajectory = self._flow.latent_to_data(latent)
        flow_trajectory.prior_logp = prior_logp
        return data, flow_trajectory
        

    def data_to_latent(self, data: VerletFrame) -> Tuple[VerletFrame, FlowTrajectoryMetadata]:
        # run flow backward
        latent, flow_trajectory = self._flow.data_to_latent(data)
        # get prior density
        prior_logp = self._prior.get_logp(data)
        flow_trajectory.prior_logp = prior_logp
        return latent, flow_trajectory

    # Training is done in the backwards direction
    def forward(self, data: VerletFrame) -> torch.Tensor:
        _, flow_metadata = self.data_to_latent(data)
        return flow_metadata.total_logp()

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

    def latent_to_data(self, data: VerletFrame) -> Tuple[VerletFrame, FlowTrajectoryMetadata]:
        """
        Computes image of data under the flow and log densities
        """
        flow_logp = torch.zeros(data.num_frames, device=data.device) 
        # Initialize trajectory and add starting frame
        flow_metadata = FlowTrajectoryMetadata()
        flow_metadata.trajectory.append(data.copy())
        # Run through flow layers
        for coupling_layer in self.coupling_layers:
            data, layer_logp = coupling_layer(data, FlowDirection.FORWARD)
            flow_logp = flow_logp + layer_logp
            flow_metadata.trajectory.append(data.copy())
        flow_metadata.flow_logp = flow_logp
        return data, flow_metadata

    def data_to_latent(self, data: VerletFrame) -> Tuple[VerletFrame, FlowTrajectoryMetadata]:
        """
        Computes the pre-image of data under the flow and the latent log densities
        """
        flow_logp = torch.zeros(data.num_frames, device=data.device) 
        # Initialize trajectory and add starting frame
        flow_metadata = FlowTrajectoryMetadata()
        flow_metadata.trajectory.append(data.copy())
        # Run through flow layers
        for coupling_layer in reversed(self.coupling_layers):
            data, layer_logp = coupling_layer(data, FlowDirection.BACKWARD)
            flow_logp = flow_logp + layer_logp
            flow_metadata.trajectory.append(data.copy())
        flow_metadata.flow_logp = flow_logp
        return data, flow_metadata

    # NEEDS TO BE UPDATED
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


class FlowDirection(Enum):
    FORWARD = 0
    BACKWARD = 1


class FrameDockingCouplingLayer(nn.Module):
    """
    SE3 Verlet coupling layer for docking
    """

    def __init__(self, device, **kwargs):
        super().__init__()
        self.st_net = FrameDockingScoreModel(**kwargs)
        # constrain timestep to be >0
        self.log_timestep = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, data: VerletFrame, flow_direction: FlowDirection):
        """
        Implements a single SE3 Verlet coupling layer

        Args:
            data: ligand/protein structures
            flow_direction: direction of flow (usually forward for inference, backwards for training)

        Returns:
            change in log densities
        """
        # compute timestep
        timestep = torch.exp(self.log_timestep)
        if flow_direction == FlowDirection.FORWARD:
            # forward coupling step
            s_rot, s_tr, t_rot, t_tr = self.st_net(data)
            data.v_rot = data.v_rot * torch.exp(s_rot) + t_rot
            data.v_tr = data.v_tr * torch.exp(s_tr) + t_tr
            update_rot = timestep * data.v_rot
            update_tr = timestep * data.v_tr
            v_rot_matrix = axis_angle_to_matrix(update_rot).squeeze()
            data = apply_update(data, v_rot_matrix, update_tr)
            delta_pxv = -torch.sum(s_rot, axis=-1) - torch.sum(s_tr, axis=-1)
            return data, delta_pxv
        elif flow_direction == FlowDirection.BACKWARD:
            # backward coupling step
            update_rot = timestep * data.v_rot
            update_tr = timestep * data.v_tr
            negative_v_rot_matrix = axis_angle_to_matrix(-update_rot).squeeze()
            data = apply_update(data, negative_v_rot_matrix, -data.v_tr)
            # all shapes batch_size x 3
            s_rot, s_tr, t_rot, t_tr = self.st_net(data)
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
        data, _ = self.forward(data, FlowDirection.FORWARD)
        # go backwards
        recreated_data, _ = self.forward(data, FlowDirection.BACKWARD)
        self.train()

        recreated_data_pos = recreated_data.detach().clone()
        recreated_data_v_rot = recreated_data.v_tr.detach().clone()
        recreated_data_v_tr = recreated_data.v_rot.detach().clone()

        print(f"Recreated v_rot: {recreated_data_v_rot}")
        print(f"Recreated v_tr: {recreated_data_v_tr}")

        assert torch.allclose(data_pos, recreated_data_pos, rtol=1e-05, atol=1e-05)
        assert torch.allclose(data_v_rot, recreated_data_v_rot, rtol=1e-05, atol=1e-05)
        assert torch.allclose(data_v_tr, recreated_data_v_tr, rtol=1e-05, atol=1e-05)
