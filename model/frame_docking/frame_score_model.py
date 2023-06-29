import torch
from e3nn import o3

from model.model_utils import GaussianSmearing

class FrameDockingScoreModel(torch.nn.Module):
    """
    receptor:       ligand:
    1               1
    |               |
    0 - 2           0 - 2
    |               |
    3               3
    
    Notes: Needs work!
    """
    def __init__(self, distance_embed_dim = 4, max_cross_offset = 5):
        super().__init__()
        self.distance_embed_dim = 5
        self.offset_distance_embedding = GaussianSmearing(
            0.0, max_cross_offset, distance_embed_dim)
        
        self.tp1 = o3.FullyConnectedTensorProduct(irreps_in1 = f'{self.distance_embed_dim}x0e + 1e', irreps_in2 = f'{distance_embed_dim}x0e + 1e', irreps_out = '20x0e + 5x1e', internal_weights = True)
        self.tp2 = o3.FullyConnectedTensorProduct(irreps_in1 = tp1.irreps_out, irreps_in2 = f'{self.distance_embed_dim}x0e + 1e', irreps_out = '9x0e + 1x1e', internal_weights = True)
                  
    def forward(self, data):
        """
        Args:
            data: ligand and receptor frames
        Returns:
            s_rot, s_tr, t_rot, t_tr: scores
        """
        # cross vector
        edge_vec = data['ligand'].pos - data['receptor'].pos
        
        # distance
        distances = torch.sqrt(torch.sum(torch.square(edge_vec[:,1:4]), axis = -1, keepdims=True))
        offset_distance_embedding = self.offset_distance_embedding(offset)
        
        # normalize edge features
        edge_vec = edge_vec / distances
        
        # masks
        num_edges = edge_vec.shape[0]
        edge_mask1 = [idx for idx in range(num_edges) if idx % 4 == 1]
        edge_mask2 = [idx for idx in range(num_edges) if idx % 4 == 2]
        edge_mask3 = [idx for idx in range(num_edges) if idx % 4 == 3]
        
        # construct {self.distance_embed_dim}x0e + 1e features
        edge_features1 = edge_vec[edge_mask1]
        edge_features2 = edge_vec[edge_mask2]
        edge_features3 = edge_vec[edge_mask3]
        
        # compute features tensor features tensor features
        output = self.tp1(edge_features1, edge_features2)
        output = self.tp2(output, edge_features3)
        # return scores
        s_rot, s_tr = output[:,:3], output[:,3:6]
        t_rot, t_tr = output[6:9], output[9:12]
        return s_rot, s_tr, t_rot, t_tr
