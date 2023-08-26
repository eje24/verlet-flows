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


def log_gaussian_density(x: torch.Tensor):
    """
    Args:
        x: tensor of shape ... x N
    Returns:
        R^N Gaussian density
    """
    return -0.5 * torch.sum(x * x, axis=-1) - 0.5 * torch.log(torch.tensor(2 * math.pi))
