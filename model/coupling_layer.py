import torch
import torch.nn as nn
from model.model import TensorProductScoreModel
from utils.geometry_utils import apply_update, axis_angle_to_matrix


class SE3CouplingLayer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.s_net = TensorProductScoreModel(device)
        self.t_net = TensorProductScoreModel(device)

    def forward(self, x, v_rot, v_tr, reverse=False):
        """
        x, HeteroGraph, current structure of x
        v, torch.Tensor, current velocity element, of the form [v_rot, v_tr]
        Returns x_prime, v_rot_prime, v_tr_prime, delta_pxv
        Satisfies log p(x_prime, v_prime) = log p(x,v) + delta_pxv
        """
        if not reverse:
            s_rot, s_tr = self.s_net(x)
            t_rot, t_tr = self.t_net(x)
            v_rot_prime = v_rot * torch.exp(s_rot) + t_rot
            v_tr_prime = v_tr * torch.exp(s_tr) + t_tr
            v_rot_prime_mat = axis_angle_to_matrix(v_rot_prime)
            x_prime = apply_update(x, v_rot_prime_mat, v_tr_prime)
            delta_pxv = -torch.sum(s_rot) - torch.sum(s_tr)
            return x_prime, v_rot_prime, v_tr_prime, delta_pxv
        else:
            # backward mapping
            x_prime = apply_update(x, axis_angle_to_matrix(-v_rot), -v_tr)
            s_rot, s_tr = self.s_net(x_prime)
            t_rot, t_tr = self.t_net(x_prime)
            v_rot_prime = (v_rot - t_rot) * torch.exp(-s_rot)
            v_tr_prime = (v_tr - t_tr) * torch.exp(-s_tr)
            delta_pxv = torch.sum(s_rot) + torch.sum(s_tr)
            return x_prime, v_rot_prime, v_tr_prime, delta_pxv
