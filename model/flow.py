import torch.nn as nn
from coupling_layer import SE3CouplingLayer


class HomogenousAugmentedNF(nn.Module):
    def __init__(self, device, num_layers=5):
        self.device = device
        self.num_layers = num_layers
        self.coupling_layers = nn.ModuleList(
            [SE3CouplingLayer(device) for _ in range(num_layers)])

    def forward(self, x, v_rot, v_tr, reverse=False):
        delta_pxv = 0
        for coupling_layer in self.coupling_layers:
            data, _delta_pxv = coupling_layer(x, v_rot, v_tr, reverse)
            delta_pxv += _delta_pxv
        return data, delta_pxv
