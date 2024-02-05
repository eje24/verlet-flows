from abc import ABC, abstractmethod
from enum import Enum
import time
import math

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchdyn.numerics import odeint
from typing import Tuple, List, Optional
import numpy as np

from model.integrator import AugmentedFlowTrajectory, build_integrator
from model.flow import AugmentedFlow
from datasets.dist import Sampleable, Density, Distribution, GMM, Gaussian, Funnel
from datasets.aug_data import AugmentedData


class AugmentedWrapper:
    def __init__(self, source: Distribution, target: Sampleable, flow: AugmentedFlow):
        self._source = source
        self._target = target
        self._flow = flow
        self._device = None

        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

    def to(self, device):
        self._source = self._source.to(device)
        self._target = self._target.to(device)
        self._flow = self._flow.to(device)
        self._device = device

    def integrate(self, data: AugmentedData, num_steps: int, integrator: str, trace_estimator: str, verbose: bool=False) -> Tuple[AugmentedData, AugmentedFlowTrajectory]:
        # Wrap flow if necessary to become compatible with integrator
        flow = self._flow.wrap_for_integration(integrator)
        # Initialize trajectory
        trajectory = AugmentedFlowTrajectory()
        trajectory.trajectory.append(data)
        trajectory.source_logp = self._source.get_log_density(data)
        # Integrate
        integrator = build_integrator(integrator)
        if integrator.name == 'numeric':
            data, trajectory = integrator.integrate(flow, data, trajectory, num_steps, reverse=False, trace_estimator=trace_estimator, verbose=verbose)
        else:
            data, trajectory = integrator.integrate(flow, data, trajectory, num_steps, reverse=False)
        return data, trajectory

    def reverse_integrate(self, data: AugmentedData, num_steps: int, integrator) -> Tuple[AugmentedData, AugmentedFlowTrajectory]:
        # Wrap flow if necessary to become compatible with integrator
        flow = self._flow.wrap_for_integration(integrator)
        # Initialize trajectory
        trajectory = AugmentedFlowTrajectory()
        trajectory.trajectory.append(data)
        # trajectory.source_logp = self._source.get_log_density(data)
        # Integrate
        integrator = build_integrator(integrator)
        data, trajectory = integrator.integrate(flow, data, trajectory, num_steps, reverse=True)
        return data, trajectory


    def sample(self, n_sample, n_steps=60, integrator='numeric', trace_estimator='autograd_trace', verbose=False) -> Tuple[AugmentedData, AugmentedFlowTrajectory]:
        source_data = self._source.sample(n_sample)
        return self.integrate(source_data, n_steps, integrator, trace_estimator=trace_estimator, verbose=verbose)

    def reverse_sample(self, n_sample, n_steps=60, integrator='numeric') -> Tuple[AugmentedData, AugmentedFlowTrajectory]:
        target_data = self._target.sample(n_sample)
        return self.reverse_integrate(target_data, n_steps, integrator)

    # Graphs the q projection of the learned flow at p=0
    @torch.no_grad()
    def graph_flow_marginals(self, n_marginals = 5, bins=25, xlim=3, ylim=3, mode='streamplot'):
        qx = np.linspace(-xlim, xlim, bins)
        qy = np.linspace(-ylim, ylim, bins)
        QX, QY = np.meshgrid(qx, qy)
        qxy = torch.tensor(np.stack([QX.reshape(-1), QY.reshape(-1)]).T, dtype=torch.float32, device='cuda')

        fix, axs = plt.subplots(1, n_marginals, figsize=(100, 20))
        t_base = torch.ones(bins ** 2, 1, device=self._device)
        p = torch.zeros_like(qxy, device=self._device)
        for t in reversed(range(n_marginals)):
            data = AugmentedData(qxy, p, t / (n_marginals - 1) * t_base)
            dq, _ = self._flow.get_flow(data)
            dq = dq.reshape(bins, bins, 2).detach().cpu().numpy()
            # NOTE: We negate dq and dp because target is at t=0 and source is at t=1
            # TODO: switch signs when we correct source to be at t=0 and target to be at t=1
            if mode == 'streamplot':
                axs[t].streamplot(QX, QY, -dq[:,:,0], -dq[:,:,1])
            elif mode == 'quiver':
                axs[t].quiver(QX, QY, -dq[:,:,0], -dq[:,:,1], scale=30.0)
            axs[t].set_aspect('equal', 'box')
            axs[t].set_title('t = ' + str(t / n_marginals))
            axs[t].set_xlim(-xlim, xlim)
            axs[t].set_ylim(-ylim, ylim)
        plt.show()

    @torch.no_grad()
    def graph_end_marginals(self, n_samples, n_integrator_steps, xlim=3, ylim=3):
        data, trajectory = self.sample(n_samples, n_integrator_steps)
        samples = trajectory.trajectory[-1].q.detach().cpu().numpy()
        plt.hist2d(samples[:,0], samples[:,1], bins=300, density=True)
        plt.title('t = 1.0')
        plt.gca().set_aspect('equal', 'box')
        plt.xlim(-xlim, xlim)
        plt.ylim(-ylim, ylim)
        plt.show()

    def get_marginal_idxs(self, n_marginals, n_integrator_steps):
        marginal_idxs = []
        for i in range(0, n_marginals - 1):
            idx = int(i / (n_marginals - 1) * n_integrator_steps)
            marginal_idxs.append(idx)
        marginal_idxs.append(n_integrator_steps)
        return marginal_idxs

    def graph_marginals(self, trajectory: AugmentedFlowTrajectory, marginal_idxs: list[int]):
        n_marginals = len(marginal_idxs)
        n_integrator_steps = len(trajectory.trajectory) - 1
        # Plot marginals
        fig, axs = plt.subplots(1, n_marginals, figsize=(10, 10))
        for i in range(n_marginals):
            idx = marginal_idxs[i]
            samples = trajectory.trajectory[idx].q.detach().cpu().numpy()
            axs[i].hist2d(samples[:,0], samples[:,1], bins=100, density=True)
            axs[i].set_aspect('equal', 'box')
            axs[i].set_title('t = ' + str(idx / n_integrator_steps))
            # axs[i].set_xlim(-xlim, xlim)
            # axs[i].set_ylim(-ylim, ylim)
        plt.subplots_adjust(wspace=1.0)
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def graph_forward_marginals(self, n_samples, n_marginals, n_integrator_steps = 60, integrator='numeric'):
        data, trajectory = self.sample(n_samples, n_integrator_steps, integrator)
        marginal_idxs = self.get_marginal_idxs(n_marginals, n_integrator_steps)
        self.graph_marginals(trajectory, marginal_idxs)

    @torch.no_grad()
    def graph_backward_marginals(self, n_samples, n_marginals, n_integrator_steps = 60, integrator='numeric'):
        data, trajectory = self.reverse_sample(n_samples, n_integrator_steps, integrator)
        marginal_idxs = self.get_marginal_idxs(n_marginals, n_integrator_steps)
        self.graph_marginals(trajectory, marginal_idxs)

    # Experiments
    def reverse_kl(self, N=10000) -> float:
        data, trajectory = self.sample(N)
        pushforward_logp = trajectory.source_logp + trajectory.flow_logp
        target_logp = self._target.get_log_density(data)
        return torch.mean(pushforward_logp - target_logp)

    @torch.no_grad()
    def estimate_z(self, n_sample=10000, n_steps=60, integrator='numeric', trace_estimator:str = 'autograd_trace', n_trial: int = 10) -> float:
        data, trajectory = self.sample(n_sample * n_trial, n_steps, integrator, trace_estimator=trace_estimator)
        pushforward_logp = trajectory.source_logp + trajectory.flow_logp
        target_logp = self._target.get_log_density(data)
        log_diff = target_logp - pushforward_logp
        z_estimates = torch.zeros((n_trial,), device=data.device)
        for i in range(n_trial):
            z_estimates[i] = torch.mean(torch.exp(log_diff[i * n_sample : (i + 1) * n_sample]))
        log_importance_weights = log_diff
        return z_estimates, log_importance_weights

    # Wrapper around Z estimation which batches calls to estimate_z
    def estimate_z_wrapper(self, n_sample=10000, n_steps=60, integrator='numeric', trace_estimator:str = 'autograd_trace', n_trial: int = 10) -> float:
        # Note: memory constraints limit us to ~100000 samples per call
        max_samples_per_call = 100000
        estimates = torch.zeros((n_trial,), device=self._device)
        trial_idx = 0
        while n_trial > 0:
            n_trial_batch = min(n_trial, max_samples_per_call // n_sample)
            n_trial -= n_trial_batch
            new_z_estimates, _ = self.estimate_z(n_sample, n_steps, integrator, trace_estimator, n_trial_batch)
            estimates[trial_idx : trial_idx + n_trial_batch] = new_z_estimates
            trial_idx += n_trial_batch
        return estimates
    


