from typing import Tuple
from collections import namedtuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from utils.distributions import uniform_so3_random
from utils.distributions import log_uniform_density_so3, log_gaussian_density

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

    def copy(self):
        return self.tensor_op("detach").tensor_op("clone")

    @property
    def device(self):
        return self.receptor.device

    @property
    def ligand_center(self):
        return torch.mean(self.ligand, axis=-2)

    @property
    def receptor_center(self):
        return torch.mean(self.receptor, axis=-2)

    @property
    def num_frames(self):
        return self.receptor.size()[0]

def gaussian_r3(mean = 0.0, std = 1.0):
    return torch.normal(mean=torch.tensor([mean, mean, mean]), std = torch.tensor([std, std, std]))

class FramePrior(Dataset):
    """
    receptor:       ligand:
    1               1
    |               |
    0 - 2           0 - 2
    |               |
    3               3
    """

    def __init__(self, device, num_items: int = 0, *args, **kwargs):
        self.num_items = num_items
        self.receptor = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.float,
            device=device,
        )
        self.ligand = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.float,
            device=device,
        )
        self.device = device
        
    def __len__(self):
        return self.num_items

    def __getitem__(self, idx: int):
        v_rot = gaussian_r3().to(self.device)
        v_tr = gaussian_r3().to(self.device)
        noise_tr = gaussian_r3().to(self.device)
        noise_rot = uniform_so3_random(1).squeeze().float().to(self.device)
        return (
            self.receptor @ noise_rot,
            (self.ligand + noise_tr) @ noise_rot,
            v_rot,
            v_tr,
        )

    def get_x_logp(self, data: VerletFrame) -> torch.Tensor:
        noise_tr = data.receptor_center - data.ligand_center
        return log_uniform_density_so3() + log_gaussian_density(noise_tr)

    def get_v_logp(self, data: VerletFrame) -> torch.Tensor:
        return log_gaussian_density(data.v_rot) + log_gaussian_density(data.v_tr)

    def get_logp(self, data: VerletFrame) -> torch.Tensor:
        return self.get_x_logp(data) + self.get_v_logp(data)

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
        v_rot = gaussian_r3().to(self.device)
        v_tr = gaussian_r3().to(self.device)
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
