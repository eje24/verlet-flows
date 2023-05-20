from scipy.stats import multivariate_normal
import torch
import numpy as np

# density of so3 is 1/8pi^2 (Haar measure)
LOG_SO3_UNIFORM_DENSITY = -np.log(8 * np.pi**2)


def log_uniform_density_so3():
    return LOG_SO3_UNIFORM_DENSITY


def log_gaussian_density_r3(x: torch.Tensor):
    return np.log(multivariate_normal.pdf(x))
