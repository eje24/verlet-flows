import torch.nn as nn
import torch
from typing import Tuple, List, Optional

import matplotlib.pyplot as plt

from datasets.dist import Sampleable, Density, GMM, Gaussian, VerletGaussian, VerletGMM
from datasets.verlet import VerletData


class FlowTrajectory:
    def __init__(self):
        self.trajectory: List[VerletData] = list()
        self.source_logp: Optional[torch.Tensor] = None
        self.flow_logp: Optional[torch.Tensor] = None


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

        return q_nvp_matrix

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

        return p_nvp_matrix


class VerletIntegrator():
    def __init__(self):
        pass

    # Returns the next state after a single step of Verlet integration, as well as the log determinant of the Jacobian of the transformation
    def integrate_step(self, flow: VerletFlow, data: VerletData, dt: float) -> Tuple[VerletData, torch.tensor]:
        dlogp = torch.zeros((data.batch_size(),), device=data.device())
        # Volume-preserving q update
        q_vp = flow.q_vp(data)
        data = VerletData(data.q + (dt / 4) * q_vp, data.p, data.t + (dt / 4))
        # Non-volume preserving q update
        q_nvp_matrix = flow.q_nvp(data)
        new_q = torch.bmm(torch.linalg.matrix_exp((dt / 4) * q_nvp_matrix), data.q.unsqueeze(2)).squeeze(2)
        data = VerletData(new_q, data.p, data.t + (dt / 4))
        dlogp -= torch.einsum('ijj->i', ((dt / 4) * q_nvp_matrix))
        # Volume-preserving p update
        p_vp = flow.p_vp(data)
        data = VerletData(data.q, data.p + (dt / 4) * p_vp, data.t + (dt / 4))
        # Non-volume preserving p update
        p_nvp_matrix = flow.p_nvp(data)
        new_p = torch.bmm(torch.linalg.matrix_exp((dt / 4) * p_nvp_matrix), data.p.unsqueeze(2)).squeeze(2)
        data = VerletData(data.q, new_p, data.t + (dt / 4))
        dlogp -= torch.einsum('ijj->i', ((dt / 4) * p_nvp_matrix))
        return data, dlogp

    # Starting from a ginen state, Verlet-integrate the given flow from t=0 to t=1 using the prescribed number of steps
    def integrate(self, flow: VerletFlow, data: VerletData, trajectory: FlowTrajectory, num_steps: int = 10) -> Tuple[VerletData, FlowTrajectory]:
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
        # Prepare trajectory
        trajectory = FlowTrajectory()
        trajectory.trajectory.append(data)
        trajectory.source_logp = self._source.get_density(data)
        trajectory.flow_logp = torch.zeros((data.batch_size(),), device=data.device())
        # Integrate
        data, trajectory = self._integrator.integrate(self._flow, data, trajectory, num_steps)
        return data, trajectory

    def forward_kl_loss(self, batch_size, num_steps) -> torch.Tensor:
        raise NotImplementedError

    def reverse_kl_loss(self, batch_size, num_steps) -> torch.Tensor:
        source_data = self._source.sample(batch_size)
        data, trajectory = self.source_to_target(source_data, num_steps)
        pushforward_logp = trajectory.source_logp + trajectory.flow_logp
        target_logp = self._target.get_density(data)
        return torch.mean(pushforward_logp - target_logp)

    # Energy-based training using the integrator
    def forward(self, batch_size, num_steps) -> Tuple[VerletData, torch.Tensor]:
        return self.reverse_kl_loss(batch_size, num_steps)

    # Non training-related functions

    # Sample from flow
    def sample(self, num_samples, num_steps) -> Tuple[VerletData, FlowTrajectory]:
        source_data = self._source.sample(num_samples)
        return self.source_to_target(source_data, num_steps)

    def graph_time_marginals(self, num_samples, num_steps):
        data, trajectory = self.sample(num_samples, num_steps)
        num_marginals = num_steps + 1
        fig, axs = plt.subplots(1, num_marginals, figsize=(10, 10))
        for i in range(num_marginals):
            samples = trajectory.trajectory[i].q.detach().cpu().numpy()
            axs[i].hist2d(samples[:,0], samples[:,1], bins=100, density=True)
            axs[i].set_aspect('equal', 'box')
            axs[i].set_title('t = ' + str(i / num_steps))
            axs[i].set_xlim(-2, 2)
            axs[i].set_ylim(-2, 2)
        plt.subplots_adjust(wspace=1.0)
        plt.show()


    def load_from_file(self, filename):
        self.load_state_dict(torch.load(filename))

    @staticmethod
    def default_gmm_flow_wrapper(args, device):
        # Initialize model
        verlet_flow = VerletFlow(2, 4, 10)

        # Initialize sampleable source distribution
        q_sampleable = Gaussian(torch.zeros(2, device=device), torch.eye(2, device=device))
        p_sampleable = Gaussian(torch.zeros(2, device=device), torch.eye(2, device=device))
        source = VerletGaussian(
            q_sampleable = q_sampleable,
            p_sampleable = p_sampleable,
        )

        # Initialize target density
        q_density = GMM(device=device, nmode=args.nmodes, xlim=0.5, scale=0.2)
        p_density = Gaussian(torch.zeros(2, device=device), torch.eye(2, device=device))
        target = VerletGMM(
            q_density = q_density,
            p_density = p_density,
        )

        # Initialize flow wrapper
        flow_wrapper = FlowWrapper(
            flow = verlet_flow,
            source = source,
            target = target,
        )
        flow_wrapper.to(device)
        
        return flow_wrapper
        







