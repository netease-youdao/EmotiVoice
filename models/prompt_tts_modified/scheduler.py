"""
This code is modified from https://github.com/alibaba-damo-academy/KAN-TTS.
"""

from torch.optim.lr_scheduler import *  
from torch.optim.lr_scheduler import _LRScheduler

class FindLR(_LRScheduler):


    def __init__(self, optimizer, max_steps, max_lr=10):
        self.max_steps = max_steps
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        return [
            base_lr
            * ((self.max_lr / base_lr) ** (self.last_epoch / (self.max_steps - 1)))
            for base_lr in self.base_lrs
        ]


class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(
            last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5)
        )
        return [base_lr * scale for base_lr in self.base_lrs]