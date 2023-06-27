from typing import Tuple
from collections import namedtuple

import torch
from torch.utils.data import Dataset, DataLoader 
from scipy.spatial.transform import Rotation as R

VerletFrame = namedtuple('VerletFrame', ['receptor', 'ligand', 'v_rot', 'v_tr'])
        
class FrameDataset(Dataset):
    """
    receptor:       ligand:
    1               1
    |               |
    0 - 2           0 - 2
    |               |
    3               3
    """
    def __init__(self, num_items: int, *args, **kwargs):
        self.num_items = num_items
        self.receptor = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
        offset = torch.tensor([[0.5,0.5,0.5]])
        self.ligand = 0.5 * self.receptor + offset
        
    def __len__(self):
        return self.num_items
    
    def __getitem__(self, idx: int):
        v_rot = torch.rand(1, 3)
        v_tr = torch.rand(1, 3)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()        
        return self.receptor @ random_rotation, self.ligand @ random_rotation, v_rot, v_tr
    
    
    def construct_loaders(self, num_items: int, *args, **kwargs) -> Tuple[DataLoader, DataLoader]:
        """
        Args:
            TODO
        Returns:
            training and validation loaders
        """
        train_dataset = FrameDataset(cache_path=args.cache_path, split_path=args.split_train, keep_original=True,
                                num_conformers=args.num_conformers)
        val_dataset = FrameDataset(cache_path=args.cache_path,
                              split_path=args.split_val, keep_original=True)
        
        loader_class = DataLoader
        train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size,
                                    num_workers=args.num_dataloader_workers, shuffle=True)
        val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_dataloader_workers, shuffle=True)

        return train_loader, val_loader        
