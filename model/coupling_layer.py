import torch
import torch.nn as nn
from model.model import TensorProductScoreModel
from utils.geometry_utils import apply_update, axis_angle_to_matrix


class SE3CouplingLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.s_net = TensorProductScoreModel(**kwargs)
        self.t_net = TensorProductScoreModel(**kwargs)

    def forward(self, data, reverse=False):
        """
        @param data, Batch, current structure of x
        @returns , delta_pxv
        Satisfies log p(x_prime, v_prime) = log p(x,v) + delta_pxv
        """
        if not reverse:
            s_rot, s_tr = self.s_net(data)
            t_rot, t_tr = self.t_net(data)
            data.v_rot = data.v_rot * torch.exp(s_rot) + t_rot
            data.v_tr = data.v_tr * torch.exp(s_tr) + t_tr
            v_rot_prime_mat = axis_angle_to_matrix(data.v_rot)
            for (i, complex) in enumerate(data.to_data_list()):
                complex = apply_update(
                    complex, v_rot_prime_mat[i], data.v_tr[i])
            delta_pxv = -torch.sum(s_rot, axis=-1) - torch.sum(s_tr, axis=-1)
            return data, delta_pxv
        else:
            # backward mapping
            for (i, complex) in enumerate(data.to_data_list()):
                complex = apply_update(complex, axis_angle_to_matrix(-data.v_rot[i]), -data.v_tr[i])
            s_rot, s_tr = self.s_net(data)
            t_rot, t_tr = self.t_net(data)
            v_rot_prime = (data.v_rot - t_rot) * torch.exp(-s_rot)
            v_tr_prime = (data.v_tr - t_tr) * torch.exp(-s_tr)
            delta_pxv = torch.sum(s_rot, axis=-1) + torch.sum(s_tr, axis=-1)
            return data, delta_pxv
