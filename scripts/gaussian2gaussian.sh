#!/usr/bin/bash
#
# Generate a random number
random_number=$RANDOM

# Call the Python script with all parameters inline
python old_train.py \
--run_name=gaussian2gaussian_$random_number \
--source=gaussian \
--target=gaussian \
--batch_size=256 \
--n_epochs=20 \
--loss=reverse_kl_loss \
--lr=0.01 \
--scheduler=plateau \

