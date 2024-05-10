# Verlet Flows
Implementation of [Verlet Flows: Exact Likelihood Integrators for Flow-Based Generative Models](https://arxiv.org/abs/2405.02805) by Ezra Erives, Bowen Jing, and Tommi Jaakkola.

Train with
```
python train.py run_name=RUN_NAME
```
Configuration is done with Hydra, and can be accomplished by editing the top level .yaml file `/conf/config.yaml`.
