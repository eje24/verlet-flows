import torch
import torch.nn as nn
from model.model import TensorProductScoreModel
from utils.geometry_utils import apply_update, axis_angle_to_matrix
from torch_geometric.data import Batch


class SE3CouplingLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.s_net = TensorProductScoreModel(**kwargs)
        self.t_net = TensorProductScoreModel(**kwargs)

    def forward(self, data: Batch, reverse:bool=False):
        """
        Implements a single SE3 Verlet coupling layer
        
        Args:
            data: ligand/protein structures
            
        Returns:
            change in log densities
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
            data.v_rot = (data.v_rot - t_rot) * torch.exp(-s_rot)
            data.v_tr = (data.v_tr - t_tr) * torch.exp(-s_tr)
            delta_pxv = -torch.sum(s_rot, axis=-1) - torch.sum(s_tr, axis=-1)
            return data, delta_pxv
