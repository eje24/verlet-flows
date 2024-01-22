#!/usr/bin/bash
#
# Generate a random number
random_number=$RANDOM

# Call the Python script with all parameters inline
python old_train.py \
--run_name=gaussian2funnel_$random_number \
--source=gaussian \
--target=funnel \
--data_dim=10 \
--n_epochs=50 \
--loss=reverse_kl_loss \
--lr=0.001 \
--scheduler=plateau \
