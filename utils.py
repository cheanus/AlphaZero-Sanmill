import torch
from torch.utils.data import Dataset
class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        if name.startswith('__'):
            return super().__getattr__(name)
        return self[name]

class SanmillDataset(Dataset):
    def __init__(self, examples):
        self.boards, self.pis, self.vs, self.periods = list(zip(*examples))
    def __len__(self):
        return len(self.periods)
    def __getitem__(self, i):
        boards = torch.tensor(self.boards[i], dtype=torch.float32)
        target_pis = torch.tensor(self.pis[i], dtype=torch.float32)
        target_vs = torch.tensor(self.vs[i], dtype=torch.float32)
        periods = torch.tensor(self.periods[i], dtype=torch.int)
        return boards, target_pis, target_vs, periods