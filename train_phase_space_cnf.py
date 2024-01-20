import os
import sys

import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger

sys.path.append('../')
from model.cnf import PhaseSpaceCNF
from utils.parsing import parse_phase_space_cnf_args

PROJECT_NAME = 'verlet-flows'

def main():
    args = parse_phase_space_cnf_args()

    # Initialize flow
    cnf = PhaseSpaceCNF(args)

    # Initialize logger
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=args.run_name,
        save_dir='workdir',
    )

    # Train
    trainer = pl.Trainer(logger=wandb_logger, min_epochs=args.num_epochs, max_epochs=args.num_epochs)
    trainer.fit(cnf)

    # Save
    trainer.save_checkpoint(os.path.join('workdir', f'{args.run_name}.ckpt'))
    

if __name__ == "__main__":
    main()
