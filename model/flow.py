import matplotlib.pyplot as plt
import math

import torch.nn as nn
import torch
from typing import Tuple, List, Optional
import numpy as np


from datasets.dist import Sampleable, Density, GMM, Gaussian, VerletGaussian, VerletGMM
from datasets.verlet import VerletData
from utils.parsing import display_args


class FlowTrajectory:
    def __init__(self):
        self.trajectory: List[VerletData] = list()
        self.source_logp: Optional[torch.Tensor] = None
        self.flow_logp: Optional[torch.Tensor] = None


# Flow architecture based on existing literature
# See Appendix E.2 in https://arxiv.org/abs/2302.00482
class VerletFlow(nn.Module):
    def __init__(self, data_dim, num_vp_hidden, num_nvp_hidden, num_vp_layers, num_nvp_layers):
        super().__init__()
        self._data_dim = data_dim
        self._num_vp_hidden = num_vp_hidden
        self._num_nvp_hidden = num_nvp_hidden

        # Initialize layers
        self._q_vp_net = self._create_net(data_dim + 1, data_dim, num_vp_hidden, num_vp_layers)
        self._q_nvp_net = self._create_net(data_dim + 1, data_dim * data_dim, num_nvp_hidden, num_nvp_layers)
        self._p_vp_net = self._create_net(data_dim + 1, data_dim, num_vp_hidden, num_vp_layers)
        self._p_nvp_net = self._create_net(data_dim + 1, data_dim * data_dim, num_nvp_hidden, num_nvp_layers)

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

    # Below functions all return the vector field contribution, as well as the log Jacobian determinant of the transformation

    # Volume preserving component of q-update
    def q_vp(self, data: VerletData):
        # Concatenate p and time
        x = torch.cat((data.p, data.t), dim=1)
        return self._q_vp_net(x)
        
    # Non-volume preserving component of q-update
    # Returns: q_nvp_matrix, q_nvp
    def q_nvp(self, data: VerletData):
        x = torch.cat((data.p, data.t), dim=1)
        # Get matrix
        q_nvp_matrix = self._q_nvp_net(x)
        # Reshape to matrix
        q_nvp_matrix = q_nvp_matrix.view(data.batch_size, self._data_dim, self._data_dim)

        return torch.clip(q_nvp_matrix, -20, 20)

    # Volume preserving component of p-update
    # Returns p_vp
    def p_vp(self, data: VerletData):
        # Concatenate q and time
        t = data.t * torch.ones((data.batch_size,1), device=data.device, dtype=torch.float32)
        x = torch.cat((data.q, data.t), dim=1)
        return self._p_vp_net(x)

    # Non-volume preserving component of p-update
    # Returns p_vp_matrix, p_vp
    def p_nvp(self, data: VerletData):
        t = data.t * torch.ones((data.batch_size,1), device=data.device, dtype=torch.float32)
        x = torch.cat((data.q, data.t), dim=1)
        # Get matrix
        p_nvp_matrix = self._p_nvp_net(x)
        # Reshape to matrix
        p_nvp_matrix = p_nvp_matrix.view(data.batch_size, self._data_dim, self._data_dim)

        return torch.clip(p_nvp_matrix, -20, 20)

    def get_flow(self, data: VerletData):
        # Get q component
        q_vp, q_nvp_matrix = self.q_vp(data), self.q_nvp(data)
        q_nvp = torch.bmm(q_nvp_matrix, data.q.unsqueeze(2)).squeeze(2)
        dq = q_vp + q_nvp
        # Get p component
        p_vp, p_nvp_matrix = self.p_vp(data), self.p_nvp(data)
        p_nvp = torch.bmm(p_nvp_matrix, data.p.unsqueeze(2)).squeeze(2)
        dp = p_vp + p_nvp
        return dq, dp


class VerletIntegrator():
    def __init__(self):
        pass

    # Returns the next state after a single step of Verlet integration, as well as the log determinant of the Jacobian of the transformation
    def integrate_step(self, flow: VerletFlow, data: VerletData, dt: float) -> Tuple[VerletData, torch.tensor]:
        dlogp = torch.zeros((data.batch_size,), device=data.device)
        # Volume-preserving q update
        q_vp = flow.q_vp(data)
        data = VerletData(data.q + dt * q_vp, data.p, data.t)
        # Non-volume-preserving q update
        q_nvp_matrix = flow.q_nvp(data)
        new_q = torch.bmm(torch.linalg.matrix_exp(dt * q_nvp_matrix), data.q.unsqueeze(2)).squeeze(2)
        data = VerletData(new_q, data.p, data.t)
        dlogp -= torch.einsum('ijj->i', dt * q_nvp_matrix)
        # Volume-preserving p update
        p_vp = flow.p_vp(data)
        data = VerletData(data.q, data.p + dt * p_vp, data.t)
        # Non-volume-preserving p update
        p_nvp_matrix = flow.p_nvp(data)
        new_p = torch.bmm(torch.linalg.matrix_exp(dt * p_nvp_matrix), data.p.unsqueeze(2)).squeeze(2)
        data = VerletData(data.q, new_p, data.t)
        dlogp -= torch.einsum('ijj->i', dt * p_nvp_matrix)
        # Time-update step
        data = VerletData(data.q, data.p, data.t + dt)
        return data, dlogp

    def reverse_integrate_step(self, flow: VerletFlow, data: VerletData, dt: float) -> Tuple[VerletData, torch.tensor]:
        dlogp = torch.zeros((data.batch_size,), device=data.device)
        # Time-update step
        data = VerletData(data.q, data.p, data.t - dt)
        # Non-volume-preserving p update
        p_nvp_matrix = flow.p_nvp(data)
        new_p = torch.bmm(torch.linalg.matrix_exp(-dt * p_nvp_matrix), data.p.unsqueeze(2)).squeeze(2)
        data = VerletData(data.q, new_p, data.t)
        dlogp -= torch.einsum('ijj->i', dt * p_nvp_matrix)
        # Volume-preserving p update
        p_vp = flow.p_vp(data)
        data = VerletData(data.q, data.p - dt * p_vp, data.t)
        # Non-volume-preserving q update
        q_nvp_matrix = flow.q_nvp(data)
        new_q = torch.bmm(torch.linalg.matrix_exp(-dt * q_nvp_matrix), data.q.unsqueeze(2)).squeeze(2)
        data = VerletData(new_q, data.p, data.t)
        dlogp -= torch.einsum('ijj->i', dt * q_nvp_matrix)
        # Volume-preserving q update
        q_vp = flow.q_vp(data)
        data = VerletData(data.q - dt * q_vp, data.p, data.t)
        return data, dlogp

    # Starting from a ginen state, Verlet-integrate the given flow from t=0 to t=1 using the prescribed number of steps
    def integrate(self, flow: VerletFlow, data: VerletData, trajectory: FlowTrajectory, num_steps: int = 10) -> Tuple[VerletData, FlowTrajectory]:
        trajectory.flow_logp = torch.zeros((data.batch_size,), device=data.device)
        dt = 1.0 / num_steps
        for _ in range(num_steps):
            data, dlogp = self.integrate_step(flow, data, dt)
            trajectory.trajectory.append(data)
            trajectory.flow_logp += dlogp
        return data, trajectory

    def reverse_integrate(self, flow: VerletFlow, data: VerletData, trajectory: FlowTrajectory, num_steps: int = 10) -> Tuple[VerletData, FlowTrajectory]:
        trajectory.flow_logp = torch.zeros((data.batch_size,), device=data.device)
        dt = 1.0 / num_steps
        for _ in range(num_steps):
            data, dlogp = self.reverse_integrate_step(flow, data, dt)
            trajectory.trajectory.append(data)
            trajectory.flow_logp += dlogp
        return data, trajectory

    # Check invertibility of integrator
    def assert_consistency(self, flow: VerletFlow, source_data: VerletData, num_steps: int = 10):
        assert torch.allclose(source_data.t, torch.zeros_like(source_data.t, device=source_data.device), atol=1e-7), f"source_data.t = {source_data.t}"
        # Perform forward integration pass
        forward_trajectory = FlowTrajectory()
        forward_trajectory.trajectory.append(source_data)
        target_data, trajectory = self.integrate(flow, source_data, forward_trajectory, num_steps)

        # Assert that target data is at time t=1
        assert math.isclose(target_data.t, 1.0), f"target_data.t = {target_data.t}"

        # Perform reverse integration pass
        reverse_trajectory = FlowTrajectory()
        reverse_trajectory.trajectory.append(target_data)
        recreated_source_data, reverse_trajectory = self.reverse_integrate(flow, target_data, reverse_trajectory, num_steps)

        # Assert that recreated source data is at time t=0
        assert torch.allclose(recreated_source_data.t, torch.zeros_like(recreated_source_data.t, device=recreated_source_data.device), atol=1e-7), f"recreated_source_data.t = {recreated_source_data.t}"
        # Assert that source data and recreated source data are equal
        assert torch.allclose(source_data.q, recreated_source_data.q, atol=1e-7), f"source_data.q = {source_data.q}, recreated_source_data.q = {recreated_source_data.q}"
        assert torch.allclose(source_data.p, recreated_source_data.p, atol=1e-7), f"source_data.p = {source_data.p}, recreated_source_data.p = {recreated_source_data.p}"
        # Assert that flow logp matches
        assert torch.allclose(forward_trajectory.flow_logp, reverse_trajectory.flow_logp, atol=1e-7), f"forward_trajectory.flow_logp = {forward_trajectory.flow_logp}, reverse_trajectory.flow_logp = {reverse_trajectory.flow_logp}"
        print("Consistency check passed")

        
# Transforms a distribution "latent" to an (unnormalized) density "data"
class FlowWrapper(nn.Module):
    def __init__(self, device, flow: VerletFlow, source: Sampleable, target: Density):
        super().__init__()
        self._flow = flow
        self._source = source
        self._target = target
        self._integrator = VerletIntegrator()
        self._device = device

    def source_to_target(self, data: VerletData, num_steps) -> Tuple[VerletData, FlowTrajectory]:
        # Prepare trajectory
        trajectory = FlowTrajectory()
        trajectory.trajectory.append(data)
        trajectory.source_logp = self._source.get_density(data)
        # Integrate
        data, trajectory = self._integrator.integrate(self._flow, data, trajectory, num_steps)
        return data, trajectory

    def sample(self, num_samples, num_steps) -> Tuple[VerletData, FlowTrajectory]:
        source_data = self._source.sample(num_samples)
        return self.source_to_target(source_data, num_steps)

    # Losses
    def forward_kl_loss(self, batch_size, num_steps) -> torch.Tensor:
        raise NotImplementedError

    def reverse_kl_loss(self, batch_size, num_steps) -> torch.Tensor:
        data, trajectory = self.sample(batch_size, num_steps)
        pushforward_logp = trajectory.source_logp + trajectory.flow_logp
        target_logp = self._target.get_density(data)
        return torch.mean(pushforward_logp - target_logp)

    def energy_loss(self, batch_size, num_steps):
        data, trajectory = self.sample(batch_size, num_steps)
        target_logp = self._target.get_density(data)
        return -torch.mean(target_logp)

    # Sample from target, integrate backwards, compute density
    # Only available if we can directly sample from target
    def reverse_energy_loss(self, batch_size, num_steps):
        # Sample from target
        data = self._target.sample(batch_size)
        data = data.set_time(torch.ones_like(data.t, device=data.device))
        # Prepare trajectory
        trajectory = FlowTrajectory()
        trajectory.trajectory.append(data)
        # Integrate backwards
        data, trajectory = self._integrator.reverse_integrate(self._flow, data, trajectory, num_steps)
        source_logp = self._source.get_density(data)
        return -torch.mean(source_logp)

    def flow_matching_loss(self, batch_size):
        # Sample from source and target
        source_data = self._source.sample(batch_size)
        target_data = self._target.sample(batch_size)
        # Interploate
        t = torch.rand((batch_size,1), device=source_data.device)
        interpolated_data = VerletData.interpolate(source_data, target_data, t)
        # Compute vector field
        dq, dp = self._flow.get_flow(interpolated_data)
        # Compute expected vector field
        expected_dq = (target_data.q - source_data.q)
        expected_dp = (target_data.p - source_data.p)
        # Compute loss
        return torch.mean((dq - expected_dq)**2 + (dp - expected_dp)**2)

    def forward(self, batch_size, num_steps) -> Tuple[VerletData, torch.Tensor]:
        # return self.energy_loss(batch_size, num_steps)
        return self.reverse_kl_loss(batch_size, num_steps)
        # return self.flow_matching_loss(batch_size)

    # Non training-related functions
    def assert_consistency(self, batch_size, num_steps):
        data = self._source.sample(batch_size)
        self._integrator.assert_consistency(self._flow, data, num_steps)

    # Graphs the q projection of the learned flow at p=0
    def graph_flow_marginals(self, num_steps = 5, bins=100, xlim=3, ylim=3):
        qx = np.linspace(-xlim, xlim, bins)
        qy = np.linspace(-ylim, ylim, bins)
        QX, QY = np.meshgrid(qx, qy)
        qxy = torch.tensor(np.stack([QX.reshape(-1), QY.reshape(-1)]).T, dtype=torch.float32, device='cuda')

        fix, axs = plt.subplots(1, num_steps, figsize=(10, 10))
        for t in range(num_steps):
            dq, _ = self._flow.get_flow(VerletData(qxy, torch.zeros_like(qxy, device=self._device), t / num_steps))
            dq = dq.reshape(bins, bins, 2).detach().cpu().numpy()
            axs[t].streamplot(QX, QY, dq[:,:,0], dq[:,:,1])
            axs[t].set_aspect('equal', 'box')
            axs[t].set_title('t = ' + str(t / num_steps))
            axs[t].set_xlim(-xlim, xlim)
            axs[t].set_ylim(-ylim, ylim)
        plt.show()

    def graph_end_marginals(self, num_samples, num_steps, xlim=-3, ylim=3):
        data, trajectory = self.sample(num_samples, num_steps)
        samples = trajectory.trajectory[-1].q.detach().cpu().numpy()
        plt.hist2d(samples[:,0], samples[:,1], bins=300, density=True)
        plt.title('t = 1.0')
        plt.gca().set_aspect('equal', 'box')
        plt.xlim(-xlim, xlim)
        plt.ylim(-ylim, ylim)
        plt.show()
        

    def graph_time_marginals(self, num_samples, num_steps, xlim=-3, ylim=3):
        data, trajectory = self.sample(num_samples, num_steps)
        num_marginals = num_steps + 1
        fig, axs = plt.subplots(1, num_marginals, figsize=(10, 10))
        for i in range(num_marginals):
            samples = trajectory.trajectory[i].q.detach().cpu().numpy()
            axs[i].hist2d(samples[:,0], samples[:,1], bins=100, density=True)
            axs[i].set_aspect('equal', 'box')
            axs[i].set_title('t = ' + str(i / num_steps))
            axs[i].set_xlim(-xlim, xlim)
            axs[i].set_ylim(-ylim, ylim)
        plt.subplots_adjust(wspace=1.0)
        plt.tight_layout()
        plt.show()


    def load_from_file(self, filename):
        self.load_state_dict(torch.load(filename))

# Utilities for generating and loading FlowWrapper objects

def build_source(args, device) -> Sampleable:
    source = None
    if args.source == 'gmm':
        q_sampleable = GMM(device=device, nmode=args.source_nmodes, xlim=1.0, scale=0.5)
        p_sampleable = Gaussian(torch.zeros(2, device=device), torch.eye(2, device=device))
        source = VerletGMM(
            q_density = q_sampleable,
            p_density = p_sampleable,
            t = 0.0
        )
    elif args.source == 'gaussian':
        q_sampleable = Gaussian(args.source_gaussian_mean + torch.zeros(2, device=device), torch.tensor([[args.source_gaussian_xvar, args.source_gaussian_xyvar], [args.source_gaussian_xyvar, args.source_gaussian_yvar]], device=device))
        p_sampleable = Gaussian(torch.zeros(2, device=device), torch.eye(2, device=device))
        source = VerletGaussian(
            q_sampleable = q_sampleable,
            p_sampleable = p_sampleable,
            t = 0.0
        )
    else:
        raise ValueError('Invalid source distribution type')
    return source

def build_target(args, device) -> Density:
    target = None
    if args.target == 'gmm':
        q_density = GMM(device=device, nmode=args.target_nmodes, xlim=1.0, scale=0.5)
        p_density = Gaussian(torch.zeros(2, device=device), torch.eye(2, device=device))
        target = VerletGMM(
            q_density = q_density,
            p_density = p_density,
            t = 1.0
        )
    elif args.target == 'gaussian':
        q_density = Gaussian(args.target_gaussian_mean + torch.zeros(2, device=device), torch.tensor([[args.target_gaussian_xvar, args.target_gaussian_xyvar], [args.target_gaussian_xyvar, args.target_gaussian_yvar]], device=device))
        p_density = Gaussian(torch.zeros(2, device=device), torch.eye(2, device=device))
        target = VerletGaussian(
            q_sampleable = q_density,
            p_sampleable = p_density,
            t = 1.0
        )
    else:
        raise ValueError('Invalid target type')
    return target


def default_flow_wrapper(args, device) -> FlowWrapper:
    # Initialize model
    verlet_flow = VerletFlow(2, args.num_vp_hidden_units, args.num_nvp_hidden_units, args.num_vp_hidden_layers, args.num_nvp_hidden_layers)

    # Initialize source and target
    source = build_source(args, device)
    target = build_target(args, device)

    # Initialize flow wrapper
    flow_wrapper = FlowWrapper(
        device = device,
        flow = verlet_flow,
        source = source,
        target = target,
    )
    flow_wrapper.to(device)
    
    return flow_wrapper

def load_saved(path) -> FlowWrapper:
    saved_dict = torch.load(path)
    args = saved_dict['args']
    display_args(args)
    flow_wrapper = default_flow_wrapper(args, args.device)
    flow_wrapper.load_state_dict(saved_dict['model'])
    return flow_wrapper

    
        







