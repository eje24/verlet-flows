import torch.nn as nn
from coupling_layer import CouplingLayer


class HomogenousAugmentedNF(nn.Module):
    def __init__(self, device, num_layers=5):
        self.device = device
        self.num_layers = num_layers
        self.coupling_layers = nn.ModuleList(
            [CouplingLayer(device) for _ in range(num_layers)])

    def forward(self, x, v, reverse=False):
        delta_pxv = 0
        for coupling_layer in self.coupling_layers:
            x, v, _delta_pxv = coupling_layer(x, v, reverse)
            delta_pxv += _delta_pxv
        return x, v, delta_pxv
