import os
import sys

import pytorch_lightning as pl

sys.path.append('../')
from model.cnf import PhaseSpaceCNF
from utils.parsing import parse_phase_space_cnf_args


def main():
    args = parse_phase_space_cnf_args()

    # Initialize flow
    cnf = PhaseSpaceCNF(args)

    # Train
    trainer = pl.Trainer(min_epochs=args.num_epochs, max_epochs=args.num_epochs)
    trainer.fit(cnf)

    # Save
    trainer.save_checkpoint(os.path.join('workdir', f'{args.run_name}.ckpt'))
    

if __name__ == "__main__":
    main()
