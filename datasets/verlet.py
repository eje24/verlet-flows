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

    def get_qp(self):
        return torch.cat([self.q, self.p], dim=1)

    def get_qpt(self):
        return torch.cat([self.q, self.p, self.t], dim=1)

    @staticmethod
    def from_qp(qp: torch.Tensor, t: float):
        dim = qp.size()[1] // 2
        q = qp[:, :dim]
        p = qp[:, dim:]
        t = t * torch.ones((qp.size()[0], 1), device=qp.device)
        return VerletData(q, p, t)
    
    @staticmethod
    def interpolate(data1: 'VerletData', data2: 'VerletData', alpha: torch.Tensor):
        return VerletData(
            q = data1.q * (1 - alpha) + data2.q * alpha,
            p = data1.p * (1 - alpha) + data2.p * alpha,
            t = data1.t * (1 - alpha) + data2.t * alpha
        )
    
    
