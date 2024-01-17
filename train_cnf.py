import os
import sys

import pytorch_lightning as pl

sys.path.append('../')
from model.cnf import CNF
from utils.parsing import parse_cnf_args


def main():
    args = parse_cnf_args()

    # Initialize flow
    cnf = CNF(args)

    # Train
    trainer = pl.Trainer(min_epochs=args.num_epochs, max_epochs=args.num_epochs)
    trainer.fit(cnf)

    # Save
    trainer.save_checkpoint(os.path.join('workdir', f'{args.run_name}.ckpt'))
    

if __name__ == "__main__":
    main()
