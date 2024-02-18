import os
import sys
import subprocess

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import hydra
import wandb
from wandb.wandb_run import Run
from omegaconf import DictConfig, OmegaConf

sys.path.append('../')
from model.cnf import AugmentedCNF
from model.flow_matching import FlowMatching

PROJECT_NAME = 'verlet-flows'

def check_uncommitted_changes():
    try:
        # Run the git status command and capture the output
        result = subprocess.run(['git', 'status', '--porcelain'], stdout=subprocess.PIPE, check=True, text=True)
        
        # If the output is not empty, there are uncommitted changes
        if result.stdout.strip():
            print("Warning: There are uncommitted changes in the repository. Please commit or stash them before running this program.")
            sys.exit(1)
        else:
            print("No uncommitted changes detected. Proceeding with the program.")
    except subprocess.CalledProcessError as e:
        # Handle errors from the git command
        print(f"Failed to check for uncommitted changes: {e}")
        sys.exit(1)
    except FileNotFoundError:
        # git command not found, likely not in a git repository or git is not installed
        print("git command not available. Make sure you are in a Git repository and git is installed.")
        sys.exit(1)

def get_current_commit_info():
    try:
        # Get the current commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        
        # Get the description (commit message) of the current commit
        commit_description = subprocess.check_output(['git', 'log', '-1', '--pretty=%B'], text=True).strip()

        return commit_hash, commit_description
    except subprocess.CalledProcessError as e:
        print(f"Error executing git command: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Git is not available. Please ensure git is installed and accessible.")
        sys.exit(1)

def get_model(cfg: DictConfig) -> pl.LightningModule:
    if cfg.training.loss == 'cnf':
        return AugmentedCNF(cfg)
    elif cfg.training.loss == 'flow_matching':
        return FlowMatching(cfg)

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # Check for uncommitted changes
    if not cfg.debug:
        check_uncommitted_changes()

    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Initialize model
    model = get_model(cfg)

    # Initialize logger
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=cfg.run_name,
        save_dir='workdir',
    )
    wandb_run = wandb_logger.experiment

    # Log commit hash and description
    if isinstance(wandb_run, Run):
        commit_hash, commit_description = get_current_commit_info()
        wandb_run.config.update({'commit_hash': commit_hash, 'commit_description': commit_description})

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

    # Train model
    trainer.fit(model)
    

if __name__ == "__main__":
    main()
