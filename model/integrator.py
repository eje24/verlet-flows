from abc import ABC, abstractmethod
from typing import Tuple
import math

import torch
from torchdiffeq import odeint
from torchdyn.models import NeuralODE, CNF as CNFWrapper

from datasets.verlet import VerletData
from model.flow import Flow, VerletFlow, TorchdynAugmentedFlowWrapper

class FlowTrajectory:
    def __init__(self):
        self.trajectory: List[VerletData] = list()
        self.source_logp: Optional[torch.Tensor] = None
        self.flow_logp: Optional[torch.Tensor] = None

class Integrator(ABC):
    pass

class NumericIntegrator(Integrator):
    def __init__(self):
        pass

    @torch.no_grad()
    def integrate(self, flow: TorchdynAugmentedFlowWrapper, data: VerletData, trajectory: FlowTrajectory, num_steps: int, reverse=False) -> VerletData:
        ode = NeuralODE(CNFWrapper(flow), sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4)
        t_span = torch.linspace(1.0, 0.0, num_steps + 1)
        # Augment data with time and concatenate logp and time
        data = data.get_qp()
        logp = torch.zeros((data.size()[0], 1), device=data.device)
        if reverse:
            # Target to sample
            t = torch.zeros((data.size()[0], 1), device=data.device)
            t_span = torch.linspace(0.0, 1.0, num_steps + 1)
        else:
            # Sample to target
            t = torch.ones((data.size()[0], 1), device=data.device)
            t_span = torch.linspace(1.0, 0.0, num_steps + 1)
        data = torch.cat([logp, t, data], dim=1)
        # Integrate
        _, traj = ode(data, t_span)
        # Put trajectory into AugmentedFlowTrajectory
        for t in range(1, num_steps + 1):
            new_data = VerletData.from_qp(traj[t, :, 2:], float(t / num_steps))
            trajectory.trajectory.append(new_data)
        # Get flow logp
        trajectory.flow_logp = traj[-1][:,0]
        return trajectory.trajectory[-1], trajectory


class VerletIntegrator(Integrator):
    def __init__(self):
        # This parameter indicates whether integrating populates the flow_logp property of the trajectory
        self.supports_likelihood = True

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

def build_integrator(integrator: str, **kwargs) -> Integrator:
    if integrator == 'verlet':
        return VerletIntegrator()
    elif integrator == 'numeric':
        return NumericIntegrator()
    else:
        raise ValueError(f"Unknown integrator {integrator}")

