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
        
    def check_invertibility(self, data):
        """
        Checks that coupling layer is invertible
        
        Args:
            data: ligand/protein structures
        """
        data_pos = data['ligand'].pos.detach().clone()
        data_v_rot = data.v_tr.detach().clone()
        data_v_tr = data.v_rot.detach().clone()
        
        print(f'Initial v_rot: {data_v_rot}')
        print(f'Initial v_tr: {data_v_tr}')
        
        self.eval()
        # go forward
        data, _ = self.forward(data, reverse=False)
        # go backwards
        recreated_data, _ = self.forward(data, reverse=True) 
        self.train()
        
        recreated_data_pos = recreated_data['ligand'].pos.detach().clone()
        recreated_data_v_rot = recreated_data.v_tr.detach().clone()
        recreated_data_v_tr = recreated_data.v_rot.detach().clone() 
        
        print(f'Recreated v_rot: {recreated_data_v_rot}')
        print(f'Recreated v_tr: {recreated_data_v_tr}')
        
        assert torch.allclose(data_pos, recreated_data_pos, rtol=1e-05, atol=1e-05)
        assert torch.allclose(data_v_rot, recreated_data_v_rot, rtol=1e-05, atol=1e-05)
        assert torch.allclose(data_v_tr, recreated_data_v_tr, rtol=1e-05, atol=1e-05)
        assert torch.allclose(log_pxv, recreated_log_pxv, rtol=1e-05, atol=1e-05)
        
