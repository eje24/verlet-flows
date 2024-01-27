import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

sys.path.append('../')
from model.cnf import AugmentedCNF
from model.flow_matching import FlowMatching

PROJECT_NAME = 'verlet-flows'

def get_model(cfg: DictConfig) -> pl.LightningModule:
    if cfg.training.loss == 'cnf':
        return AugmentedCNF(cfg)
    elif cfg.training.loss == 'flow_matching':
        return FlowMatching(cfg)

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    # Initialize model
    model = get_model(cfg)

    # Initialize logger
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=cfg.run_name,
        save_dir='workdir',
    )

    # Initialize checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('workdir', cfg.run_name),
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        every_n_epochs=cfg.training.val_every_n_epochs,
        mode='min',
    )

    # Initialize trainer
    trainer = pl.Trainer(logger=wandb_logger, 
                         min_epochs=cfg.training.num_epochs, 
                         max_epochs=cfg.training.num_epochs, 
                         check_val_every_n_epoch=cfg.training.val_every_n_epochs,
                         callbacks=[checkpoint_callback])
    trainer.fit(model)
    

if __name__ == "__main__":
    main()
