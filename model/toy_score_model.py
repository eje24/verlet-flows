import torch
from e3nn import o3

from models.model_utils import GaussianSmearing
from utils.geometric_utils import matrix_to_axis_angle

class FrameDockingScoreModel(torch.nn.Module):
    def __init__(self, distance_embed_dim = 4, max_cross_offset = 5):
        super().__init__()
        self.distance_embed_dim = 4
        self.offset_distance_embedding = GaussianSmearing(
            0.0, max_cross_offset, distance_embed_dim)
        
        self.tp1 = o3.FullTensorProduct(irreps_in1 = '1e', irreps_in2 = '1e', internal_weights = True)
        self.tp2 = o3.FullTensorProduct(irreps_in1 = tp1.irreps_out, irreps_in2 = '1e', internal_weights = True)
        self.tp3 = o3.FullyConnectedTensorProduct(irreps_in1 = tp2.irreps_out, irreps_in2 = f'{3 + distance_embed_dim}x0e', irreps_out = '9x0e + 1e', internal_weights = True)
                  
    def forward(self, data):
        """
        receptor:       ligand:
        1               1
        |               |
        0 - 2           0 - 2
        |               |
        3               3
        """
        # cross vectors
        cross_vec_1 = data['ligand'].pos[...,1,:] - data['receptor'].pos[...,0,:]
        cross_vec_2 = data['ligand'].pos[...,2,:] - data['receptor'].pos[...,0,:]
        cross_vec_3 = data['ligand'].pos[...,3,:] - data['receptor'].pos[...,0,:]
        
        # distance
        offset = torch.mean(data['ligand'].pos - data['receptor'].pos, axis = -2)
        offset_distance_embedding = self.offset_distance_embedding(offset)
        
        # orientation
        ligand_edges = data['ligand'].pos[...,1:,:] - data['ligand'].pos[...,:1,:]
        receptor_edges = data['receptor'].pos[...,1:,:] - data['receptor'].pos[...,:1,:]
        relative_edges = ligand_edges @ receptor_edges
        orientation = matrix_to_axis_angle(relative_edges)
        
        # concatenate distance embedding and orientation scalars
        scalar_features = torch.cat([offset_distance_embedding, orientation], axis=-1)
        
        output = self.tp1(cross_vec1, cross_vec2)
        output = self.tp2(output, cross_vec3)
        output = self.tp3(output, scalar_features)
        
        s_rot, s_tr = output[:,:3], output[:,3:6]
        t_rot, t_tr = output[6:9], output[9:12]
        return s_rot, s_tr, t_rot, t_tr
