import torch
import torch.utils.data as data
import pytorch_lightning as pl
from omegaconf import DictConfig

from datasets.dist import Sampleable, build_augmented_distribution
from datasets.aug_data import AugmentedData
from model.flow import AugmentedFlow, build_augmented_flow
from model.wrapper import AugmentedWrapper
from model.ot import augmented_emd_reorder

class FlowMatchingDataset(data.Dataset):
    def __init__(self, source: Sampleable, target: Sampleable, size: int):
        super().__init__()
        self._source = source
        self._target = target
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        source_data = self._source.sample(1).get_qp()
        target_data = self._target.sample(1).get_qp()
        time = torch.rand((1,)).to(source_data.device)
        return source_data.squeeze(0), target_data.squeeze(0), time

class FlowMatching(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Initialize source, target, train
        self.source = build_augmented_distribution(self.cfg.source, self.device, 1.0)
        self.target = build_augmented_distribution(self.cfg.target, self.device, 0.0)
        self.train_set = FlowMatchingDataset(self.source, self.target, self.cfg.training.num_train)
        self.val_set = FlowMatchingDataset(self.source, self.target, self.cfg.training.num_val)
        
        # Initialize model
        self.flow = build_augmented_flow(self.cfg.flow, self.target)


    def setup(self, stage):
        self.source.to(self.device)
        self.target.to(self.device)
        self.train_losses = []
        self.val_losses = []

        # Initialize source, target, train
        self.train_loader = data.DataLoader(self.train_set, batch_size=self.cfg.training.batch_size, shuffle=True)
        self.val_loader = data.DataLoader(self.val_set, batch_size=self.cfg.training.batch_size, shuffle=True)

    def compute_flow_matching_loss(self, batch):
        # Unpack
        source_data, target_data, time = batch
        # print(source_data.shape, target_data.shape, time.shape)
        source_data = AugmentedData.from_qp(source_data, 1.0)
        target_data = AugmentedData.from_qp(target_data, 0.0)
        # Reorder to minimize earth mover's distance
        # print(source_data.q.shape, source_data.p.shape, target_data.q.shape, target_data.p.shape)
        source_data, target_data = augmented_emd_reorder(source_data, target_data)
        # Interpolate
        interpolated_data = AugmentedData.interpolate(source_data, target_data, time)
        # Compute vector field
        dq, dp = self.flow.get_flow(interpolated_data)
        # Compute expected vector field
        # NOTE: we do source - target because source is at time 1.0 and target is at time 0.0
        expected_dq = (source_data.q - target_data.q)
        expected_dp = (source_data.p - target_data.p)
        return torch.mean((dq - expected_dq)**2 + (dp - expected_dp)**2)
        

    def training_step(self, batch, batch_idx):
        loss = self.compute_flow_matching_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_losses.append(loss)
        return {'loss': loss}

    def on_train_epoch_end(self):
        # Log training loss
        avg_loss = torch.stack(self.train_losses).mean()
        self.train_losses = []
        print(f"Device {self.device} | Epoch {self.current_epoch} | Train Loss: {avg_loss}")

    def validation_step(self, batch, batch_idx):
        loss = self.compute_flow_matching_loss(batch)
        self.val_losses.append(loss)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        # Log validation loss
        avg_loss = torch.stack(self.val_losses).mean()
        self.val_losses = []
        print(f"Device {self.device} | Epoch {self.current_epoch} | Validation Loss: {avg_loss}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.flow.parameters(), lr=0.01)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def export_to_wrapper(self):
        return AugmentedWrapper(self.source, self.target, self.flow)
