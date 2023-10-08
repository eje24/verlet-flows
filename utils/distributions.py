import torch
import math

from scipy.spatial.transform import Rotation as R

# density of so3 is 1/8pi^2 (Haar measure)
_LOG_SO3_UNIFORM_DENSITY = -torch.log(torch.tensor(8 * math.pi**2))


######################
#   Uniform SO(3)    #
######################


def log_uniform_density_so3():
    return _LOG_SO3_UNIFORM_DENSITY


def uniform_so3_random(num_rotations: int):
    return torch.from_numpy(R.random(num_rotations).as_matrix())


####################
#     Gaussian     #
####################


# Density helper function for simple Gaussians
def log_gaussian_density(x: torch.Tensor, mean = torch.tensor(0.0), std = torch.tensor(1.0)):
    """
    Args:
        x: tensor of shape ... x N
    Returns:
        R^N Gaussian density
    """
    N = x.shape[-1]
    return -N/2 * torch.log(torch.tensor(2 * math.pi)) - N / 2 * torch.log(std) - 1 / (2 * std) * ((x - mean) @ (x - mean).T)
