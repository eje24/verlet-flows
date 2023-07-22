# Based on https://github.com/gcorso/DiffDock/blob/main/models/score_model.py
import math

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean
import numpy as np
from e3nn.nn import BatchNorm

from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims
from models.model_utils import GaussianSmearing, TensorProductConvLayer


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims, lm_embedding_type=None):
        # first element of feature_dims tuple is a list with the length of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1]
        self.lm_embedding_type = lm_embedding_type
        # create an embedding for each categorical feature
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        if self.lm_embedding_type is not None:
            if self.lm_embedding_type == "esm":
                self.lm_embedding_dim = 1280
            else:
                raise ValueError(
                    "LM Embedding type was not correctly determined. LM embedding type: ",
                    self.lm_embedding_type,
                )
            self.lm_embedding_layer = torch.nn.Linear(
                self.lm_embedding_dim + emb_dim, emb_dim
            )

    def forward(self, x):
        x_embedding = 0
        if self.lm_embedding_type is not None:
            assert (
                x.shape[1]
                == self.num_categorical_features
                + self.num_scalar_features
                + self.lm_embedding_dim
            )
        else:
            assert (
                x.shape[1] == self.num_categorical_features + self.num_scalar_features
            )
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(
                x[
                    :,
                    self.num_categorical_features : self.num_categorical_features
                    + self.num_scalar_features,
                ]
            )
        if self.lm_embedding_type is not None:
            x_embedding = self.lm_embedding_layer(
                torch.cat([x_embedding, x[:, -self.lm_embedding_dim :]], axis=1)
            )
        return x_embedding


class DockingScoreModel(torch.nn.Module):
    def __init__(
        self,
        in_lig_edge_features=4,
        sh_lmax=2,
        ns=16,
        nv=4,
        num_conv_layers=2,
        lig_max_radius=5,
        rec_max_radius=30,
        cross_max_distance=250,
        center_max_distance=30,
        distance_embed_dim=32,
        cross_distance_embed_dim=32,
        use_second_order_repr=False,
        batch_norm=True,
        dynamic_max_cross=False,
        dropout=0.0,
        lm_embedding_type=None,
    ):
        super(TensorProductScoreModel, self).__init__()
        self.in_lig_edge_features = in_lig_edge_features
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.num_conv_layers = num_conv_layers

        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims)
        self.lig_edge_embedding = nn.Sequential(
            nn.Linear(in_lig_edge_features + distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        self.rec_node_embedding = AtomEncoder(
            emb_dim=ns,
            feature_dims=rec_residue_feature_dims,
            lm_embedding_type=lm_embedding_type,
        )
        self.rec_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        self.cross_edge_embedding = nn.Sequential(
            nn.Linear(cross_distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        self.lig_distance_expansion = GaussianSmearing(
            0.0, lig_max_radius, distance_embed_dim
        )
        self.rec_distance_expansion = GaussianSmearing(
            0.0, rec_max_radius, distance_embed_dim
        )
        self.cross_distance_expansion = GaussianSmearing(
            0.0, cross_max_distance, cross_distance_embed_dim
        )

        if use_second_order_repr:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o + {nv}x2e",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o",
            ]
        else:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o",
                f"{ns}x0e + {nv}x1o + {nv}x1e",
                f"{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o",
            ]

        (
            lig_conv_layers,
            rec_conv_layers,
            lig_to_rec_conv_layers,
            rec_to_lig_conv_layers,
        ) = ([], [], [], [])
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                "in_irreps": in_irreps,
                "sh_irreps": self.sh_irreps,
                "out_irreps": out_irreps,
                "n_edge_features": 3 * ns,
                "hidden_features": 3 * ns,
                "residual": False,
                "batch_norm": batch_norm,
                "dropout": dropout,
            }

            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            rec_layer = TensorProductConvLayer(**parameters)
            rec_conv_layers.append(rec_layer)
            lig_to_rec_layer = TensorProductConvLayer(**parameters)
            lig_to_rec_conv_layers.append(lig_to_rec_layer)
            rec_to_lig_layer = TensorProductConvLayer(**parameters)
            rec_to_lig_conv_layers.append(rec_to_lig_layer)

        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.rec_conv_layers = nn.ModuleList(rec_conv_layers)
        self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)
        self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)

        # center of mass translation and rotation components
        self.center_distance_expansion = GaussianSmearing(
            0.0, center_max_distance, distance_embed_dim
        )
        self.center_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        self.final_conv = TensorProductConvLayer(
            in_irreps=self.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f"2x1o + 2x1e",
            n_edge_features=2 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    def forward(self, data):
        # build ligand graph
        (
            lig_node_attr,
            lig_edge_index,
            lig_edge_attr,
            lig_edge_sh,
        ) = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # build receptor graph
        (
            rec_node_attr,
            rec_edge_index,
            rec_edge_attr,
            rec_edge_sh,
        ) = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(
            data, cross_cutoff
        )
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        for l in range(len(self.lig_conv_layers)):
            # intra graph message passing
            lig_edge_attr_ = torch.cat(
                [
                    lig_edge_attr,
                    lig_node_attr[lig_src, : self.ns],
                    lig_node_attr[lig_dst, : self.ns],
                ],
                -1,
            )
            lig_intra_update = self.lig_conv_layers[l](
                lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh
            )

            # inter graph message passing
            rec_to_lig_edge_attr_ = torch.cat(
                [
                    cross_edge_attr,
                    lig_node_attr[cross_lig, : self.ns],
                    rec_node_attr[cross_rec, : self.ns],
                ],
                -1,
            )
            lig_inter_update = self.rec_to_lig_conv_layers[l](
                rec_node_attr,
                cross_edge_index,
                rec_to_lig_edge_attr_,
                cross_edge_sh,
                out_nodes=lig_node_attr.shape[0],
            )

            if l != len(self.lig_conv_layers) - 1:
                rec_edge_attr_ = torch.cat(
                    [
                        rec_edge_attr,
                        rec_node_attr[rec_src, : self.ns],
                        rec_node_attr[rec_dst, : self.ns],
                    ],
                    -1,
                )
                rec_intra_update = self.rec_conv_layers[l](
                    rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh
                )

                lig_to_rec_edge_attr_ = torch.cat(
                    [
                        cross_edge_attr,
                        lig_node_attr[cross_lig, : self.ns],
                        rec_node_attr[cross_rec, : self.ns],
                    ],
                    -1,
                )
                rec_inter_update = self.lig_to_rec_conv_layers[l](
                    lig_node_attr,
                    torch.flip(cross_edge_index, dims=[0]),
                    lig_to_rec_edge_attr_,
                    cross_edge_sh,
                    out_nodes=rec_node_attr.shape[0],
                )

            # padding original features
            lig_node_attr = F.pad(
                lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1])
            )

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

            if l != len(self.lig_conv_layers) - 1:
                rec_node_attr = F.pad(
                    rec_node_attr,
                    (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]),
                )
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update

        # compute translational and rotational score vectors
        (
            center_edge_index,
            center_edge_attr,
            center_edge_sh,
        ) = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat(
            [center_edge_attr, lig_node_attr[center_edge_index[1], : self.ns]], -1
        )
        global_pred = self.final_conv(
            lig_node_attr,
            center_edge_index,
            center_edge_attr,
            center_edge_sh,
            out_nodes=data.num_graphs,
        )

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        return tr_pred, rot_pred

    def build_lig_conv_graph(self, data):
        # builds the ligand graph edges and initial node and edge features

        # compute edges
        radius_edges = radius_graph(
            data["ligand"].pos, self.lig_max_radius, data["ligand"].batch
        )
        edge_index = torch.cat(
            [data["ligand", "ligand"].edge_index, radius_edges], 1
        ).long()
        edge_attr = torch.cat(
            [
                data["ligand", "ligand"].edge_attr,
                torch.zeros(
                    radius_edges.shape[-1],
                    self.in_lig_edge_features,
                    device=data["ligand"].x.device,
                ),
            ],
            0,
        )

        # compute initial features
        node_attr = data["ligand"].x

        src, dst = edge_index
        edge_vec = data["ligand"].pos[dst.long()] - data["ligand"].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )

        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        # builds the receptor initial node and edge embeddings
        node_attr = data["receptor"].x

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data["receptor", "receptor"].edge_index
        src, dst = edge_index
        edge_vec = data["receptor"].pos[dst.long()] - data["receptor"].pos[src.long()]

        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = edge_length_emb
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(
                data["receptor"].pos / cross_distance_cutoff[data["receptor"].batch],
                data["ligand"].pos / cross_distance_cutoff[data["ligand"].batch],
                1,
                data["receptor"].batch,
                data["ligand"].batch,
                max_num_neighbors=10000,
            )
        else:
            edge_index = radius(
                data["receptor"].pos,
                data["ligand"].pos,
                cross_distance_cutoff,
                data["receptor"].batch,
                data["ligand"].batch,
                max_num_neighbors=10000,
            )

        src, dst = edge_index
        edge_vec = data["receptor"].pos[dst.long()] - data["ligand"].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = edge_length_emb
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )

        return edge_index, edge_attr, edge_sh

    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = torch.cat(
            [
                data["ligand"].batch.unsqueeze(0),
                torch.arange(len(data["ligand"].batch))
                .to(data["ligand"].x.device)
                .unsqueeze(0),
            ],
            dim=0,
        )

        center_pos, count = torch.zeros((data.num_graphs, 3)).to(
            data["ligand"].x.device
        ), torch.zeros((data.num_graphs, 3)).to(data["ligand"].x.device)
        center_pos.index_add_(0, index=data["ligand"].batch, source=data["ligand"].pos)
        center_pos = center_pos / torch.bincount(data["ligand"].batch).unsqueeze(1)

        edge_vec = data["ligand"].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data["ligand", "ligand"].edge_index[:, data["ligand"].edge_mask].long()
        bond_pos = (data["ligand"].pos[bonds[0]] + data["ligand"].pos[bonds[1]]) / 2
        bond_batch = data["ligand"].batch[bonds[0]]
        edge_index = radius(
            data["ligand"].pos,
            bond_pos,
            self.lig_max_radius,
            batch_x=data["ligand"].batch,
            batch_y=bond_batch,
        )

        edge_vec = data["ligand"].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )

        return bonds, edge_index, edge_attr, edge_sh


# for distance embedings
class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
