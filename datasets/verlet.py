from collections import namedtuple

_VerletData = namedtuple('VerletData', 'q p t')

class VerletData(_VerletData):
    def batch_size(self) -> int:
        return self.q.size()[0]
    def device(self):
        return self.q.device
    def set_time(self, t: float):
        return VerletData(self.q, self.p, t)
