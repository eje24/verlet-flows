from collections import namedtuple

import torch

_VerletData = namedtuple('VerletData', 'q p t')

class VerletData(_VerletData):
    @property
    def batch_size(self) -> int:
        return self.q.size()[0]

    @property
    def device(self):
        return self.q.device

    def set_time(self, t: float):
        return VerletData(self.q, self.p, t)
    
    @staticmethod
    def interpolate(data1: 'VerletData', data2: 'VerletData', alpha: torch.Tensor):
        return VerletData(
            q = data1.q * (1 - alpha) + data2.q * alpha,
            p = data1.p * (1 - alpha) + data2.p * alpha,
            t = data1.t * (1 - alpha) + data2.t * alpha
        )
    
