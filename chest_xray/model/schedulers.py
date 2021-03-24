from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.optim.optimizer import Optimizer

__all__ = [
    'CosineDecayWithWarmupScheduler',
]


# Inspired by https://github.com/seominseok0429/pytorch-warmup-cosine-lr/blob/master/warmup_scheduler/scheduler.py
class CosineDecayWithWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_epochs: int,
        initial_div_factor: float = 25,
        warmup_epochs: int = 5,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None:
        initial_lr = max_lr / initial_div_factor
        self.warmup_factor = max_lr / initial_lr
        self.warmup_epochs = warmup_epochs
        self.cosine_decay = CosineAnnealingLR(optimizer, total_epochs - warmup_epochs)
        self.warmup_completed = False
        if last_epoch == -1:
            for idx, group in enumerate(optimizer.param_groups):
                group['initial_lr'] = initial_lr
                group['max_lr'] = max_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            if not self.warmup_completed:
                self.cosine_decay.base_lrs = [base_lr * self.warmup_factor for base_lr in self.base_lrs]
                self.warmup_completed = True
            return self.cosine_decay.get_lr()

        return [
            base_lr * ((self.warmup_factor - 1.) * self.last_epoch / self.warmup_epochs + 1.)
            for base_lr
            in self.base_lrs
        ]

    def step(self, epoch: int = None):
        if self.warmup_completed:
            if epoch is None:
                self.cosine_decay.step(None)
            else:
                self.cosine_decay.step(epoch - self.warmup_epochs)
            self._last_lr = self.cosine_decay.get_last_lr()
        else:
            super().step(epoch)

