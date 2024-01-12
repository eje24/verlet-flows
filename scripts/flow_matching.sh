#!/usr/bin/bash
#
# Generate a random number
random_number=$RANDOM

# Call the Python script with all parameters inline
python train.py \
--run_name=flow_matching_$random_number \
--source=gaussian \
--target=gaussian \
--loss=flow_matching_loss \
--verlet \
--n_epochs=50 \
--batch_size=64 \
--lr=0.01 \
--scheduler=plateau \
