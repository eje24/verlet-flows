from abc import ABC, abstractmethod
from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from omegaconf import DictConfig

from datasets.aug_data import AugmentedData

class AugmentedFlow(ABC):
    @abstractmethod
    def get_flow(self, data: AugmentedData) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    # Allows for numerical integration of the flow
    @abstractmethod
    def forward(self, t: float, qp: torch.Tensor) -> torch.Tensor:
        pass

    def _create_net(self, in_dims, out_dims, num_hidden_units, num_hidden_layers):
        # Contruct sequence of dimensions
        dim_list = [in_dims] + [num_hidden_units for _ in range(num_hidden_layers)] + [out_dims]
        # Construct network layers
        net_list = []
        for i in range(len(dim_list) - 1):
            curr_dim, next_dim = dim_list[i], dim_list[i+1]
            net_list.append(nn.Linear(curr_dim, next_dim))
            # Don't add a SELU after the last linear layer
            if i < len(dim_list) - 2:
                net_list.append(nn.SELU())
        return nn.Sequential(*net_list)

class NonVerletFlow(AugmentedFlow, nn.Module):
    def __init__(self, data_dim, num_hidden, num_layers):
        super().__init__()
        self._data_dim = data_dim
        self._num_hidden = num_hidden
        self._num_layers = num_layers

        # Initialize layers
        self._net = self._create_net(2 * data_dim + 1, 2 * data_dim, num_hidden, num_layers)

    def get_flow(self, data: AugmentedData):
        # Concatenate q and time
        qpt = data.get_qpt()
        d_qp = self._net(qpt)
        d_q, d_p = d_qp[:, :self._data_dim], d_qp[:, self._data_dim:]
        return d_q, d_p

    def forward(self, t: float, qp: torch.Tensor):
        # Get data in tensor format
        t = t * torch.ones((qp.size()[0], 1), device=qp.device)
        qpt = torch.cat([qp, t], dim=1)
        return self._net(qpt)

# Adds time as an input to the network at each layer, as in FFJORD
class TimeInjectionNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_hidden, num_layers):
        super().__init__()
        # Initialize modules
        module_list = []
        module_list.append(nn.Linear(in_dim + 1, num_hidden))
        for _ in range(num_layers - 1):
            module_list.append(nn.Linear(num_hidden + 1, num_hidden))
        module_list.append(nn.Linear(num_hidden + 1, out_dim))
        self._layers = nn.ModuleList(module_list)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # Concatenate q and time
        for idx, layer in enumerate(self._layers):
            x = torch.cat((x, t), dim=1)
            x = layer(x)
            if idx < len(self._layers) - 1:
                x = F.selu(x)
        return x

class NonVerletTimeFlow(nn.Module):
    def __init__(self, data_dim, num_hidden, num_layers):
        super().__init__()
        self._data_dim = data_dim
        self._net = TimeInjectionNet(2 * data_dim, 2 * data_dim, num_hidden, num_layers)

    def get_flow(self, data: AugmentedData):
        # Concatenate q and time
        d_qp = self.forward(data)
        d_q, d_p = d_qp[:, :self._data_dim], d_qp[:, self._data_dim:]
        return d_q, d_p

    def forward(self, data: AugmentedData):
        # Concatenate q and time
        x = data.get_qp()
        t = data.t
        return self._net(x, t)

    def wrap_for_integration(self, integrator: str):
        if integrator == 'verlet':
            raise ValueError('NonVerletTimeFlow cannot be used with Verlet integrator')
        elif integrator == 'numeric':
            return TorchdynAugmentedFlowWrapper(self)

# Flow architecture based on existing literature
# See Appendix E.2 in https://arxiv.org/abs/2302.00482
class VerletFlow(AugmentedFlow, nn.Module):
    def __init__(self, data_dim, num_vp_hidden, num_nvp_hidden, num_vp_layers, num_nvp_layers):
        super().__init__()
        self._data_dim = data_dim
        self._num_vp_hidden = num_vp_hidden
        self._num_nvp_hidden = num_nvp_hidden

        # Initialize layers
        self._q_vp_net = TimeInjectionNet(data_dim, data_dim, num_vp_hidden, num_vp_layers)
        self._q_nvp_net = TimeInjectionNet(data_dim, data_dim ** 2, num_nvp_hidden, num_nvp_layers)
        self._p_vp_net = TimeInjectionNet(data_dim, data_dim, num_vp_hidden, num_vp_layers)
        self._p_nvp_net = TimeInjectionNet(data_dim, data_dim ** 2, num_nvp_hidden, num_nvp_layers)
    # Below functions all return the vector field contribution, as well as the log Jacobian determinant of the transformation

    # Volume preserving component of q-update
    def q_vp(self, data: AugmentedData):
        return self._q_vp_net(data.p, data.t)
        
    # Non-volume preserving component of q-update
    # Returns: q_nvp_matrix, q_nvp
    def q_nvp(self, data: AugmentedData):
        # Get matrix
        q_nvp_matrix = self._q_nvp_net(data.p, data.t)
        # Reshape to matrix
        q_nvp_matrix = q_nvp_matrix.view(data.batch_size, self._data_dim, self._data_dim)

        return torch.clip(q_nvp_matrix, -20, 20)

    # Volume preserving component of p-update
    # Returns p_vp
    def p_vp(self, data: AugmentedData):
        return self._p_vp_net(data.q, data.t)

    # Non-volume preserving component of p-update
    # Returns p_vp_matrix, p_vp
    def p_nvp(self, data: AugmentedData):
        # Get matrix
        p_nvp_matrix = self._p_nvp_net(data.q, data.t)
        # Reshape to matrix
        p_nvp_matrix = p_nvp_matrix.view(data.batch_size, self._data_dim, self._data_dim)

        return torch.clip(p_nvp_matrix, -20, 20)

    def get_flow(self, data: AugmentedData):
        # Get q component
        q_vp, q_nvp_matrix = self.q_vp(data), self.q_nvp(data)
        q_nvp = torch.bmm(q_nvp_matrix, data.q.unsqueeze(2)).squeeze(2)
        dq = q_vp + q_nvp
        # Get p component
        p_vp, p_nvp_matrix = self.p_vp(data), self.p_nvp(data)
        p_nvp = torch.bmm(p_nvp_matrix, data.p.unsqueeze(2)).squeeze(2)
        dp = p_vp + p_nvp
        return dq, dp

    def forward(self, t: float, qp: torch.Tensor):
        data = AugmentedData.from_qp(qp, t)
        dq, dp = self.get_flow(data)
        return torch.cat((dq, dp), dim=1)

    def wrap_for_integration(self, integrator: str):
        if integrator == 'verlet':
            return self
        elif integrator == 'numeric':
            return TorchdynAugmentedFlowWrapper(self)

# Flow Wrappers
class TorchdynAugmentedFlowWrapper(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, x):
        t = x[:, :1]
        q = x[:, 1:3]
        p = x[:, 3:]
        data = AugmentedData(q, p, t)
        dq, dp = self.flow.get_flow(data)
        dt = torch.ones_like(t).to(x)
        return torch.cat([dt, dq, dp], dim=1)

def build_augmented_flow(cfg: DictConfig) -> AugmentedFlow:
    if cfg.flow_type == 'verlet':
        flow = VerletFlow(data_dim=cfg.dim,
                             num_vp_hidden=cfg.num_vp_hidden,
                             num_nvp_hidden=cfg.num_nvp_hidden,
                             num_vp_layers=cfg.num_vp_layers,
                             num_nvp_layers=cfg.num_nvp_layers)
    elif cfg.flow_type == 'non_verlet':
        flow = NonVerletTimeFlow(data_dim=cfg.dim, 
                                 num_hidden=cfg.num_hidden_units, 
                                 num_layers=cfg.num_layers)
    else:
        raise ValueError(f'Invalid flow type: {cfg.flow_type}')
    return flow
