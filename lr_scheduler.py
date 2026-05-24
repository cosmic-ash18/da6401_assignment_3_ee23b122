"""
lr_scheduler.py — Noam Learning Rate Scheduler
Made in the paper
Formula:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""

import torch
import torch.optim as optim
# use LRScheduler from torch.optim
# it's PyTorch's base class for the learning rate during training
from torch.optim.lr_scheduler import LRScheduler


# creates the custom lr scheduler
# it inherits from LRScheduler so pytorch knows how to call it during training
class NoamScheduler(LRScheduler):
    """
    Noam learning rate scheduler as described in "Attention Is All You Need".

    The optimizer's base LR is multiplied by:
        d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Use optimizer lr=1.0 for the exact paper formula.


    Sample usage:
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
    optimizer.step()
    scheduler.step()
    """

    # initialize the scheduler
    def __init__(
        self,
        optimizer: optim.Optimizer, # the adam optimizer whose LR we want to control
        d_model: int, # transformer embedding/model size
        warmup_steps: int, # number of warmup steps
        last_epoch: int = -1, # scheduler bookkeeping variable
    ) -> None:
        if d_model <= 0:
            # prevent nonsense values
            raise ValueError("d model gotta be positive")
        if warmup_steps <= 0:
            raise ValueError("warmup steps gotta be positive too")
        # save inside the object for other methods to be used
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        # tells pytorch this scheduler belongs to this optimizer
        super().__init__(optimizer, last_epoch=last_epoch)

    # the main method that computes the Noam scaling factors
    # self.last_epoch is torch's internal counter
    def _get_lr_scale(self):
        # LRScheduler performs an initial step during __init__, making last_epoch=0.
        # Therefore step=last_epoch+1 gives the first usable LR at step 1.

        # max with 1 makes the step at least 1
        step = max(1, self.last_epoch + 1)
        # compute the noam stuff
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5),)

    # pytorch calls this to get the new LR
    def get_lr(self) -> list[float]:
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]

# just a helper used to simulate and record LRs
# to plot the LR curve
def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
) -> list[float]:
    """
    Simulate the LR trajectory of NoamScheduler for `total_steps` steps.
    """
    dummy_model = torch.nn.Linear(1, 1)
    # use dumy model to get model parameters
    optimizer = optim.Adam(dummy_model.parameters(), lr=1.0)
    # this dummy optimizer has Noam scheduling
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    # store LR over time
    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # return the history
    return history

# import and plot using matplotlib
# this block only runs if we execute this file directly
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL = 512  # transformer dimension
    WARMUP_STEPS = 4000 # warmup
    TOTAL_STEPS = 20_000 # plot steps

    # simulate the scheduler and return all LR values
    lrs = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)

    # plot red vertical line to mark the warmup boundary
    # LR rises till step 4000 and then falls off inversely with sqr root of steps
    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(WARMUP_STEPS, color="red", linestyle="--", label=f"warmup={WARMUP_STEPS}")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule  (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.show()
