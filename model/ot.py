import ot
import torch

from datasets.verlet import VerletData


def get_emd_matrix(a, b, xs, xt):
    M = ot.dist(xs, xt)
    M /= M.max()
    emd_matrix = ot.emd(a, b, M)
    return emd_matrix

def fix_points(xs, xt, emd_matrix):
    perm = torch.nonzero(emd_matrix)[:,1]
    return xs, xt[perm]

# Permutes to minimize Earth Mover's Distance
def emd_reorder(xs, xt):
    n = xs.shape[0]
    a, b = torch.ones(n) / n, torch.ones(n) / n
    emd_matrix = get_emd_matrix(a, b, xs, xt)
    return fix_points(xs, xt, emd_matrix)

def verlet_emd_reorder(data_s, data_t):
    xs = data_s.get_qp()
    xt = data_t.get_qp()
    xsc, xtc = emd_reorder(xs, xt)
    return VerletData.from_qp(xsc, data_s.t), VerletData.from_qp(xtc, data_t.t)

