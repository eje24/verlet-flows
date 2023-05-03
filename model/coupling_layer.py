import torch
import torch.nn as nn
from model import TensorProductScoreModel
from utils.geometry_utils import apply_update


class CouplingLayer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.s_net = TensorProductScoreModel(device)
        self.t_net = TensorProductScoreModel(device)

    def forward(self, x, v, reverse=False):
        """
        x, HeteroGraph, current structure of x
        v, torch.Tensor, current velocity element, of the form [v_rot, v_tr]
        Returns x_prime, v_prime, delta_pxv
        Satisfies log p(x_prime, v_prime) = log p(x,v) + delta_pxv
        """
        if not reverse:
            s_rot, s_tr = self.s_net(x)
            t_rot, t_tr = self.t_net(x)
            s = torch.cat([s_rot, s_tr], dim=-1)
            t = torch.cat([t_rot, t_tr], dim=-1)
            v_prime = v * torch.exp(s) + t
            x_prime = apply_update(x, v_prime)
            delta_pxv = -torch.sum(s)
            return x_prime, v_prime, delta_pxv
        else:
            # backward mapping
            x_prime = apply_update(x, -v)
            s_rot, s_tr = self.s_net(x_prime)
            t_rot, t_tr = self.t_net(x_prime)
            s = torch.cat([s_rot, s_tr], dim=-1)
            t = torch.cat([t_rot, t_tr], dim=-1)
            v_prime = (v_prime - t) * torch.exp(-s)
            delta_pxv = torch.sum(s)
            return x_prime, v_prime, delta_pxv
