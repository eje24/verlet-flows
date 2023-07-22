import numpy as np
from tqdm import tqdm
from datasets.rigid_frames import VerletFrame

import torch


def loss_function(log_pxv):
    return -log_pxv


class AverageMeter:
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += (
                    v.sum() if self.unpooled_metrics else v
                )
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(
                    0, interval_idx[type_idx], torch.ones(len(v))
                )
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(
                        0, interval_idx[type_idx], v
                    )

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out["int" + str(i) + "_" + k] = (
                        list(self.acc.values())[type_idx][i] / self.count[type_idx][i]
                    ).item()
            return out


def train_epoch(model, loader, optimizer, device):
    model.train()
    meter = AverageMeter(["loss"])

    for receptor, ligand, v_rot, v_tr in tqdm(loader, total=len(loader)):
        optimizer.zero_grad()
        try:
            data = VerletFrame(receptor, ligand, v_rot, v_tr)
            log_pxv = model(data)
            loss = -torch.mean(log_pxv)
            loss.backward()
            optimizer.step()
            ema_weigths.update(model.parameters())
            meter.add([loss.cpu().detach()])
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif "Input mismatch" in str(e):
                print("| WARNING: weird torch_cluster error, skipping batch")
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    return meter.summary()


def test_epoch(model, loader):
    model.eval()
    meter = AverageMeter(["loss"], unpooled_metrics=True)
    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                log_pxv = model(VerletFrame(*data))
            loss = -torch.mean(log_pxv)
            meter.add([loss.cpu().detach()])

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif "Input mismatch" in str(e):
                print("| WARNING: weird torch_cluster error, skipping batch")
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    out = meter.summary()
    return out
