from abc import ABC, abstractmethod
from typing import Tuple, Callable
import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from omegaconf import DictConfig

from datasets.aug_data import AugmentedData
from datasets.dist import Distribution
from model.time_embeddings import TimeConder

class AugmentedFlow(ABC):
    @abstractmethod
    def get_flow(self, data: AugmentedData) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    # Allows for numerical integration of the flow
    @abstractmethod
    def forward(self, t: float, qp: torch.Tensor) -> torch.Tensor:
        pass


# Adds time as an input to the network at each layer, as in FFJORD
class TimeInjectionNet(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 num_hidden, 
                 num_layers):
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

class NonVerletFlow(nn.Module):
    def __init__(self, 
                 data_dim, 
                 num_hidden, 
                 num_layers, 
                 use_grad = True, 
                 log_grad_fn: Callable = None):
        super().__init__()
        self._data_dim = data_dim
        self._flow_net = TimeInjectionNet(2 * data_dim, 2 * data_dim, num_hidden, num_layers)
        self.use_grad = use_grad
        if self.use_grad:
            self.log_grad_fn = log_grad_fn
            self._grad_net = TimeConder(64, 1, 3)

    def get_flow(self, data: AugmentedData):
        # Concatenate q and time
        d_qp = self.forward(data)
        d_q, d_p = d_qp[:, :self._data_dim], d_qp[:, self._data_dim:]
        return d_q, d_p

    def forward(self, data: AugmentedData):
        # Concatenate q and time
        x = data.get_qp()
        t = data.t
        if self.use_grad:
            return self._flow_net(x, t) + self._grad_net(t) * self.log_grad_fn(data)
        else:
            return self._flow_net(x, t)

    def wrap_for_integration(self, integrator: str):
        if integrator == 'verlet':
            raise ValueError('NonVerletFlow cannot be used with Verlet integrator')
        elif integrator == 'numeric':
            return TorchdynAugmentedFlowWrapper(self)

class VerletTermType(Enum):
    Q = 0
    P = 1

    def __eq__(self, other):
        return self.value == other.value

class TaylorVerletFlowTerm(nn.Module, ABC):
    def __init__(self, term_type: VerletTermType, data_dim: int):
        super().__init__()
        self._term_type = term_type
        self._data_dim = data_dim

    @abstractmethod
    def get_flow_contribution(self, data: AugmentedData) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    # Returns the next state after a single step of Verlet integration, as well as the log determinant of the Jacobian of the transformation
    @abstractmethod
    def integrate_step(self, data: AugmentedData, dt: float) -> Tuple[AugmentedData, torch.Tensor]:
        pass

    # Returns the next state after a single step of reverse Verlet integration, as well as the log determinant of the Jacobian of the transformation
    @abstractmethod
    def reverse_integrate_step(self, data: AugmentedData, dt: float) -> AugmentedData:
        pass

class DenseOrderZeroTerm(TaylorVerletFlowTerm):
    def __init__(self, term_type: VerletTermType, data_dim: int, num_hidden: int, num_layers: int):
        super().__init__(term_type, data_dim)
        self._flow_net = TimeInjectionNet(data_dim, data_dim, num_hidden, num_layers)

    def get_flow_contribution(self, data: AugmentedData):
        if self._term_type == VerletTermType.Q:
            contribution = self._flow_net(data.p, data.t)
            return contribution, torch.zeros_like(contribution).to(contribution.device)
        else:
            contribution = self._flow_net(data.q, data.t)
            return torch.zeros_like(contribution).to(contribution.device), contribution

    def integrate_step(self, data: AugmentedData, dt: float) -> Tuple[AugmentedData, torch.Tensor]:
        dq, dp = self.get_flow_contribution(data)
        new_data = AugmentedData(data.q + dt * dq, data.p + dt * dp, data.t)
        dlogp = torch.zeros((data.batch_size,), device=data.device)
        return new_data, dlogp

    def reverse_integrate_step(self, data: AugmentedData, dt: float) -> AugmentedData:
        dq, dp = self.get_flow_contribution(data)
        new_data = AugmentedData(data.q - dt * dq, data.p - dt * dp, data.t)
        dlogp = torch.zeros((data.batch_size,), device=data.device)
        return new_data, dlogp

class DenseOrderOneTerm(TaylorVerletFlowTerm):
    def __init__(self, term_type: VerletTermType, data_dim: int, num_hidden: int, num_layers: int):
        super().__init__(term_type, data_dim)
        self._flow_net = TimeInjectionNet(data_dim, data_dim ** 2, num_hidden, num_layers)

    def get_dense_matrix(self, data: AugmentedData):
        if self._term_type == VerletTermType.Q:
            dense_matrix = self._flow_net(data.p, data.t)
        elif self._term_type == VerletTermType.P:
            dense_matrix = self._flow_net(data.q, data.t)
        else:
            raise ValueError('Invalid term type')
        dense_matrix = dense_matrix.view(data.batch_size, self._data_dim, self._data_dim)
        dense_matrix = torch.clip(dense_matrix, -20, 20)
        return dense_matrix

    def get_flow_contribution(self, data: AugmentedData):
        dense_matrix = self.get_dense_matrix(data)
        if self._term_type == VerletTermType.Q:
            contribution = torch.bmm(dense_matrix , data.q.unsqueeze(2)).squeeze(2)
            return contribution, torch.zeros_like(contribution).to(contribution.device)
        elif self._term_type == VerletTermType.P:
            contribution = torch.bmm(dense_matrix , data.p.unsqueeze(2)).squeeze(2)
            return torch.zeros_like(contribution).to(contribution.device), contribution
        else:
            raise ValueError('Invalid term type')

    def integrate_step(self, data: AugmentedData, dt: float) -> Tuple[AugmentedData, torch.Tensor]:
        dense_matrix = self.get_dense_matrix(data)
        new_data = None
        dlogp = None
        if self._term_type == VerletTermType.Q:
            # Compute new data
            new_q = torch.bmm(torch.linalg.matrix_exp(dt * dense_matrix), data.q.unsqueeze(2)).squeeze(2)
            new_data = AugmentedData(new_q, data.p, data.t)
            # Compute dlogp
            dlogp = torch.einsum('ijj->i', dt * dense_matrix)
        elif self._term_type == VerletTermType.P:
            # Compute new data
            new_p = torch.bmm(torch.linalg.matrix_exp(dt * dense_matrix), data.p.unsqueeze(2)).squeeze(2)
            new_data = AugmentedData(data.q, new_p, data.t)
            # Compute dlogp
            dlogp = torch.einsum('ijj->i', dt * dense_matrix)
        else:
            raise ValueError('Invalid term type')
        return new_data, dlogp

    def reverse_integrate_step(self, data: AugmentedData, dt: float) -> Tuple[AugmentedData, torch.Tensor]:
        dense_matrix = self.get_dense_matrix(data)
        new_data = None
        dlogp = None
        if self._term_type == VerletTermType.Q:
            # Compute new data
            new_q = torch.bmm(torch.linalg.matrix_exp(-dt * dense_matrix), data.q.unsqueeze(2)).squeeze(2)
            new_data = AugmentedData(new_q, data.p, data.t)
            # Compute dlogp
            dlogp = torch.einsum('ijj->i', dt * dense_matrix)
        elif self._term_type == VerletTermType.P:
            # Compute new data
            new_p = torch.bmm(torch.linalg.matrix_exp(-dt * dense_matrix), data.p.unsqueeze(2)).squeeze(2)
            new_data = AugmentedData(data.q, new_p, data.t)
            # Compute dlogp
            dlogp = torch.einsum('ijj->i', dt * dense_matrix)
        else:
            raise ValueError('Invalid term type')
        return new_data, dlogp

class SparseOrderNTerm(TaylorVerletFlowTerm):
    def __init__(self, term_type: VerletTermType, term_order: int, data_dim: int, num_hidden: int, num_layers: int):
        raise NotImplementedError('SparseOrderNTerm not yet implemented')

class TaylorVerletFlow(AugmentedFlow, nn.Module):
    def __init__(self,
                 data_dim: int,
                 num_hidden: int,
                 num_layers: int,
                 order: int):
        super().__init__()
        self._data_dim = data_dim
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._order = order

        self._layers = nn.ModuleDict()

        # Initialize layers 0 and 1
        self._layers['q0'] = DenseOrderZeroTerm(VerletTermType.Q, data_dim, num_hidden, num_layers)
        self._layers['p0'] = DenseOrderZeroTerm(VerletTermType.P, data_dim, num_hidden, num_layers)
        self._layers['q1'] = DenseOrderOneTerm(VerletTermType.Q, data_dim, num_hidden, num_layers)
        self._layers['p1'] = DenseOrderOneTerm(VerletTermType.P, data_dim, num_hidden, num_layers)

        # Initialize layers 2, ..., self.order
        for layer_idx in range(2, self._order + 1):
            self._layers[f'q{layer_idx}'] = SparseOrderNTerm(VerletTermType.Q, layer_idx, data_dim, num_hidden, num_layers)
            self._layers[f'p{layer_idx}'] = SparseOrderNTerm(VerletTermType.P, layer_idx, data_dim, num_hidden, num_layers)

    @property
    def order(self):
        return self._order

    @property
    def layers(self):
        layers = []
        for layer_idx in range(self._order + 1):
            layers.append(self._layers[f'q{layer_idx}'])
            layers.append(self._layers[f'p{layer_idx}'])
        return layers

    def get_flow(self, data: AugmentedData):
        dq = torch.zeros_like(data.q).to(data.q.device)
        dp = torch.zeros_like(data.p).to(data.p.device)

        for layer in self.layers:
            ddq, ddp = layer.get_flow_contribution(data)
            dq = dq + ddq
            dp = dp + ddp

        return dq, dp

    def forward(self, t: float, qp: torch.Tensor):
        data = AugmentedData.from_qp(qp, t)
        dq, dp = self.get_flow(data)
        return torch.cat((dq, dp), dim=1)

    def wrap_for_integration(self, integrator: str):
        if integrator == 'verlet':
            return self
        elif integrator == 'taylor_verlet':
            return self
        elif integrator == 'numeric':
            return TorchdynAugmentedFlowWrapper(self)

    # The following methods allow order <= 1 TaylorVerlet flows to be integrated using the Verlet integrator
    def q_vp(self, data: AugmentedData):
        return self._layers['q0'].get_flow_contribution(data)[0]

    def q_nvp(self, data: AugmentedData):
        return self._layers['q1'].get_dense_matrix(data)

    def p_vp(self, data: AugmentedData):
        return self._layers['p0'].get_flow_contribution(data)[1]

    def p_nvp(self, data: AugmentedData):
        return self._layers['p1'].get_dense_matrix(data)



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
        elif integrator == 'taylor_verlet':
            raise ValueError('VerletFlow cannot be used with Taylor-Verlet integrator')
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

def build_augmented_flow(flow_cfg: DictConfig, target: Distribution) -> AugmentedFlow:
    if flow_cfg.flow_type == 'verlet':
        flow = VerletFlow(data_dim=flow_cfg.dim,
                             num_vp_hidden=flow_cfg.num_vp_hidden,
                             num_nvp_hidden=flow_cfg.num_nvp_hidden,
                             num_vp_layers=flow_cfg.num_vp_layers,
                             num_nvp_layers=flow_cfg.num_nvp_layers)
    elif flow_cfg.flow_type == 'taylor_verlet':
        flow = TaylorVerletFlow(data_dim=flow_cfg.dim,
                             num_hidden=flow_cfg.num_hidden,
                             num_layers=flow_cfg.num_layers,
                             order=flow_cfg.order)
    elif flow_cfg.flow_type == 'non_verlet':
        log_grad_fn = target.log_grad_fn if flow_cfg.use_grad else None
        flow = NonVerletFlow(data_dim=flow_cfg.dim, 
                             num_hidden=flow_cfg.num_hidden_units, 
                             num_layers=flow_cfg.num_layers,
                             use_grad=flow_cfg.use_grad,
                             log_grad_fn=log_grad_fn
                             )
    else:
        raise ValueError(f'Invalid flow type: {flow_cfg.flow_type}')
    return flow
