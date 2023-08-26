from typing import Tuple
from collections import namedtuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from utils.distributions import uniform_so3_random

VerletFrame = namedtuple("VerletFrame", ["receptor", "ligand", "v_rot", "v_tr"])


@dataclass
class VerletFrame:
    receptor: torch.Tensor
    ligand: torch.Tensor
    v_rot: torch.Tensor
    v_tr: torch.Tensor

    def tensor_op(self, tensor_op):
        def get_attr_op_from_key(key):
            attr = getattr(self, key)
            attr_op = getattr(attr, tensor_op)
            return attr_op

        post_op_attrs = {
            key: get_attr_op_from_key(key)() for key in self.__dict__.keys()
        }

        return VerletFrame(**post_op_attrs)

    @property
    def ligand_center(self):
        return torch.mean(self.ligand, axis=-2)

    @property
    def receptor_center(self):
        return torch.mean(self.receptor, axis=-2)

    @property
    def num_frames(self):
        return self.receptor.size[0]


class FrameDataset(Dataset):
    """
    receptor:       ligand:
    1               1
    |               |
    0 - 2           0 - 2
    |               |
    3               3
    """

    def __init__(self, num_items: int, device, *args, **kwargs):
        self.num_items = num_items
        self.receptor = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.float,
            device=device,
        )
        offset = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float, device=device)
        self.ligand = 0.5 * self.receptor + offset
        self.device = device

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx: int):
        v_rot = torch.rand(3, device=self.device)
        v_tr = torch.rand(3, device=self.device)
        random_rotation = uniform_so3_random(1).squeeze().float().to(self.device)
        return (
            self.receptor @ random_rotation,
            self.ligand @ random_rotation,
            v_rot,
            v_tr,
        )

    def construct_train_loaders(args, device) -> Tuple[DataLoader, DataLoader]:
        """
        Args:
            num_train: size of training dataset
            num_val: size of validation dataset
            device: device
        Returns:
            training and validation loaders
        """
        train_dataset = FrameDataset(num_items=args.num_train, device=device)
        val_dataset = FrameDataset(num_items=args.num_val, device=device)
        loader_class = DataLoader
        train_loader = loader_class(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers,
            shuffle=True,
        )
        val_loader = loader_class(
            dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers,
            shuffle=True,
        )

        return train_loader, val_loader
