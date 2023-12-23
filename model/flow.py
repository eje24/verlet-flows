import torch.nn as nn
import torch
from typing import Tuple, List, Optional

from datasets.dist import Sampleable, Density
from datasets.verlet import VerletData


class FlowTrajectory:
    def __init__(self):
        self.trajectory: List[VerletData] = list()
        self.flow_logp: Optional[torch.Tensor] = None
        self.target_logp: Optional[torch.Tensor] = None

    def total_logp(self) -> Optional[float]:
        return self.flow_logp + self.target_logp


# Flow architecture based on existing literature
# See Appendix E.2 in https://arxiv.org/abs/2302.00482
class VerletFlow(nn.Module):
    def __init__(self, data_dim, num_vp_layers, num_nvp_layers):
        super().__init__()
        self._data_dim = data_dim
        self._num_vp_layers = num_vp_layers
        self._num_nvp_layers = num_nvp_layers

        # Initialize layers
        self._q_vp_net = nn.Sequential(nn.Linear(data_dim + 1, self._num_vp_layers),
                                       nn.SELU(),
                                       nn.Linear(self._num_vp_layers, self._num_vp_layers),
                                       nn.SELU(),
                                       nn.Linear(self._num_vp_layers, self._num_vp_layers),
                                       nn.SELU(),
                                       nn.Linear(self._num_vp_layers, data_dim))
        self._q_nvp_net = nn.Sequential(nn.Linear(data_dim + 1, self._num_nvp_layers),
                                       nn.SELU(),
                                       nn.Linear(self._num_nvp_layers, self._num_nvp_layers),
                                       nn.SELU(),
                                       nn.Linear(self._num_nvp_layers, self._num_nvp_layers),
                                       nn.SELU(),
                                       nn.Linear(self._num_nvp_layers, data_dim * data_dim))
        self._p_vp_net = nn.Sequential(nn.Linear(data_dim + 1, self._num_vp_layers),
                                       nn.SELU(),
                                       nn.Linear(self._num_vp_layers, self._num_vp_layers),
                                       nn.SELU(),
                                       nn.Linear(self._num_vp_layers, self._num_vp_layers),
                                       nn.SELU(),
                                       nn.Linear(self._num_vp_layers, data_dim))
        self._p_nvp_net = nn.Sequential(nn.Linear(data_dim + 1, self._num_nvp_layers),
                                       nn.SELU(),
                                       nn.Linear(self._num_nvp_layers, self._num_nvp_layers),
                                       nn.SELU(),
                                       nn.Linear(self._num_nvp_layers, self._num_nvp_layers),
                                       nn.SELU(),
                                       nn.Linear(self._num_nvp_layers, data_dim * data_dim))



    # Below functions all return the vector field contribution, as well as the log Jacobian determinant of the transformation

    # Volume preserving component of q-update
    def q_vp(self, data: VerletData):
        # Concatenate p and time
        t = data.t * torch.ones((data.batch_size(),1), device=data.device(), dtype=torch.float32)
        x = torch.cat((data.p, t), dim=1)
        return self._q_vp_net(x)
        
    # Non-volume preserving component of q-update
    # Returns: q_nvp_matrix, q_nvp
    def q_nvp(self, data: VerletData):
        t = data.t * torch.ones((data.batch_size(),1), device=data.device(), dtype=torch.float32)
        x = torch.cat((data.p, t), dim=1)
        # Get matrix
        q_nvp_matrix = self._q_nvp_net(x)
        # Reshape to matrix
        q_nvp_matrix = q_nvp_matrix.view(data.batch_size(), self._data_dim, self._data_dim)
        # Matrix-multiply with q
        q_nvp = torch.bmm(q_nvp_matrix, data.q.unsqueeze(2)).squeeze(2)

        return q_nvp_matrix, q_nvp

    # Volume preserving component of p-update
    # Returns p_vp
    def p_vp(self, data: VerletData):
        # Concatenate q and time
        t = data.t * torch.ones((data.batch_size(),1), device=data.device(), dtype=torch.float32)
        x = torch.cat((data.q, t), dim=1)
        return self._p_vp_net(x)

    # Non-volume preserving component of p-update
    # Returns p_vp_matrix, p_vp
    def p_nvp(self, data: VerletData):
        t = data.t * torch.ones((data.batch_size(),1), device=data.device(), dtype=torch.float32)
        x = torch.cat((data.q, t), dim=1)
        # Get matrix
        p_nvp_matrix = self._p_nvp_net(x)
        # Reshape to matrix
        p_nvp_matrix = p_nvp_matrix.view(data.batch_size(), self._data_dim, self._data_dim)
        # Multiply with p
        p_nvp = torch.bmm(p_nvp_matrix, data.p.unsqueeze(2)).squeeze(2)

        return p_nvp_matrix, p_nvp


class VerletIntegrator(nn.Module):
    def __init__(self):
        super().__init__()

    # Returns the next state after a single step of Verlet integration, as well as the log determinant of the Jacobian of the transformation
    def integrate_step(self, flow: VerletFlow, data: VerletData, dt: float) -> Tuple[VerletData, torch.tensor]:
        dlogp = torch.zeros((data.batch_size(),), device=data.device())
        # Volume-preserving q update
        q_vp = flow.q_vp(data)
        data = VerletData(data.q + (dt / 4) * q_vp, data.p, data.t + (dt / 4))
        # Non-volume preserving q update
        q_nvp_matrix, q_nvp = flow.q_nvp(data)
        data = VerletData(data.q + (dt / 4) * q_nvp, data.p, data.t + (dt / 4))
        dlogp -= torch.exp(torch.einsum('ijj->i', ((dt / 4) * q_nvp_matrix)))
        # Volume-preserving p update
        q_vp = flow.p_vp(data)
        data = VerletData(data.q, data.p + (dt / 4) * q_vp, data.t + (dt / 4))
        # Non-volume preserving p update
        p_nvp_matrix, p_nvp = flow.p_nvp(data)
        data = VerletData(data.q, data.p + (dt / 4) * q_nvp, data.t + (dt / 4))
        dlogp -= torch.exp(torch.einsum('ijj->i', ((dt / 4) * p_nvp_matrix)))
        return data, dlogp

    # Starting from a ginen state, Verlet-integrate the given flow from t=0 to t=1 using the prescribed number of steps
    def integrate(self, flow: VerletFlow, data: VerletData, num_steps: int = 10) -> Tuple[VerletData, FlowTrajectory]:
        trajectory = FlowTrajectory()
        trajectory.trajectory.append(VerletData(data.q, data.p, 0.0))
        trajectory.flow_logp = torch.zeros((data.batch_size(),), device=data.device())
        dt = 1.0 / num_steps
        for _ in range(num_steps):
            data, dlogp = self.integrate_step(flow, data, dt)
            trajectory.trajectory.append(data)
            trajectory.flow_logp += dlogp
        return data, trajectory
        
# Transforms a distribution "latent" to an (unnormalized) density "data"
class FlowWrapper(nn.Module):
    def __init__(self, flow: VerletFlow, source: Sampleable, target: Density):
        super().__init__()
        self._flow = flow
        self._source = source
        self._target = target
        self._integrator = VerletIntegrator()

    def source_to_target(self, data: VerletData, num_steps) -> Tuple[VerletData, FlowTrajectory]:
        # Run flow
        data, trajectory = self._integrator.integrate(self._flow, data, num_steps)
        # Here, we are slightly imprecise, as the density is not logp, but logp + logZ, where Z is the partition function
        trajectory.target_logp = self._target.get_density(data)
        return data, trajectory
        
    # Simulation-based training using the integrator
    # NOTE: can also train using flow-matching
    def forward(self, batch_size, num_steps) -> Tuple[VerletData, torch.Tensor]:
        source_data = self._source.sample(batch_size)
        _, trajectory = self.source_to_target(source_data, num_steps)
        return trajectory.total_logp()

