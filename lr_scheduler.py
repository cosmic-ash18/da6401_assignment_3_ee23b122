"""
lr_scheduler.py — Noam Learning Rate Scheduler
Reference: "Attention Is All You Need" (Vaswani et al., 2017)

Formula:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    """
    Noam learning rate scheduler as described in "Attention Is All You Need".

    The optimizer's base LR is multiplied by:
        d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Use optimizer lr=1.0 if you want the exact paper formula.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_lr_scale(self) -> float:
        # LRScheduler performs an initial step during __init__, making last_epoch=0.
        # Therefore step=last_epoch+1 gives the first usable LR at step 1.
        step = max(1, self.last_epoch + 1)
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5),
        )

    def get_lr(self) -> list[float]:
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
) -> list[float]:
    """
    Simulate the LR trajectory of NoamScheduler for `total_steps` steps.
    """
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    return history


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS = 20_000

    lrs = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)

    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(WARMUP_STEPS, color="red", linestyle="--", label=f"warmup={WARMUP_STEPS}")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule  (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.show()
