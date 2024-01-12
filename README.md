# Exact Likelihood Integrators for Continuous Normalizing Flows
Train with
```
python train.py --run_name=RUN_NAME ...
```
For exmple, a Verlet flow between a Gaussian and a trimodal Gaussian mixture can be trained by running 
```
python train.py --run_name=test --source=gaussian --target=gmm --target_nmodes=3 --loss=reverse_kl_loss --batch_size=256 --n_epochs=50
```
Preconfigured/templated training scripts can be found [here](https://github.com/eje24/verlet-flows/tree/main/scripts). A complete list of training arguments can be found [here](https://github.com/eje24/verlet-flows/blob/main/utils/parsing.py). Once you've trained your model, follow along this [Jupyter notebook](https://github.com/eje24/verlet-flows/blob/main/notebooks/visualization.ipynb).
