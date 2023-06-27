import torch
import math

# density of so3 is 1/8pi^2 (Haar measure)
_LOG_SO3_UNIFORM_DENSITY = -torch.log(torch.tensor(8 * math.pi**2))


def log_uniform_density_so3():
    return _LOG_SO3_UNIFORM_DENSITY


def log_gaussian_density_r3(x: torch.Tensor):
    return -0.5 * torch.sum(x * x, axis=-1) - 0.5 * torch.log(torch.tensor(2 * math.pi))