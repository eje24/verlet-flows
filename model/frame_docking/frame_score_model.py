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

    def __init__(self, distance_embed_dim=4, max_cross_offset=5):
        super().__init__()
        self._distance_embed_dim = distance_embed_dim
        self._offset_distance_embedding = GaussianSmearing(
            0.0, max_cross_offset, distance_embed_dim
        )

        self._tp1 = o3.FullyConnectedTensorProduct(
            irreps_in1=f"{self._distance_embed_dim}x0e + 1e",
            irreps_in2=f"{distance_embed_dim}x0e + 1e",
            irreps_out="20x0e + 5x1e",
            internal_weights=True,
        )
        self._tp2 = o3.FullyConnectedTensorProduct(
            irreps_in1=self._tp1.irreps_out,
            irreps_in2=f"{self._distance_embed_dim}x0e + 1e",
            irreps_out="9x0e + 1x1e",
            internal_weights=True,
        )

    def forward(self, data):
        """
        Args:
            data: ligand and receptor frames
        Returns:
            s_rot, s_tr, t_rot, t_tr: scores
        """
        # cross vector - shape batch x 4 x 3
        print(f"ligand shape: {data.ligand.shape}")
        print(f"receptor shape: {data.receptor.shape}")
        edge_vec = data.ligand - data.receptor
        print(f"edge_vec shape: {edge_vec.shape}")

        # distances - shape batch x 4 x 1
        distances = torch.sqrt(
            torch.sum(torch.square(edge_vec), axis=-1, keepdims=True)
        )
        print(f"distances shape: {distances.shape}")
        # distance_embeddings - shape batch x 4 x self._distance_embed_dim
        distance_embeddings = self._offset_distance_embedding(distances, batched=True)
        print(f"distance_embeddings shape: {distance_embeddings.shape}")

        # normalize edge features
        edge_vec = edge_vec / distances

        # construct {self._distance_embed_dim}x0e + 1e features
        edge_features1 = torch.cat([distance_embeddings[:, 1], edge_vec[:, 1]], axis=-1)
        edge_features2 = torch.cat([distance_embeddings[:, 2], edge_vec[:, 2]], axis=-1)
        edge_features3 = torch.cat([distance_embeddings[:, 3], edge_vec[:, 3]], axis=-1)

        print(f"edge_features shape: {edge_features1.shape}")

        # compute features tensor features tensor features
        output = self._tp1(edge_features1, edge_features2)
        output = self._tp2(output, edge_features3)
        # return scores - importantly, only t_tr is equivariant
        s_rot, s_tr = output[:, :3], output[:, 3:6]
        t_rot, t_tr = output[:, 6:9], output[:, 9:12]
        return s_rot, s_tr, t_rot, t_tr
