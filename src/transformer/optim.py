"""Optimization utilities for Transformer models."""

import torch
import torch.nn as nn


class NoamScheduler:
    """
    Learning rate scheduler from "Attention Is All You Need" (Noam scheme).

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    This increases the learning rate linearly for the first warmup_steps,
    then decreases it proportionally to the inverse square root of the step number.

    Args:
        optimizer: The optimizer to schedule
        d_model: Model dimension (embedding size)
        warmup_steps: Number of warmup steps (default: 4000)
        factor: Scaling factor for the learning rate (default: 1.0)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._rate = 0.0

    def step(self) -> None:
        """Update the learning rate and step the optimizer."""
        self._step += 1
        rate = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = rate
        self._rate = rate

    def get_lr(self) -> float:
        """Compute the learning rate for the current step."""
        step = self._step
        return self.factor * (
            self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

    def zero_grad(self) -> None:
        """Zero the gradients of the optimizer."""
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        """Access optimizer param groups."""
        return self.optimizer.param_groups

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "step": self._step,
            "rate": self._rate,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint."""
        self._step = state_dict["step"]
        self._rate = state_dict["rate"]


def create_noam_optimizer(
    model: nn.Module,
    d_model: int,
    warmup_steps: int = 4000,
    factor: float = 1.0,
    betas: tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-9,
) -> NoamScheduler:
    """
    Create an Adam optimizer with Noam learning rate schedule.

    Uses the hyperparameters from the original paper:
    - betas = (0.9, 0.98)
    - eps = 1e-9

    Args:
        model: The model to optimize
        d_model: Model dimension (embedding size)
        warmup_steps: Number of warmup steps (default: 4000)
        factor: Scaling factor for the learning rate (default: 1.0)
        betas: Adam beta parameters (default: (0.9, 0.98))
        eps: Adam epsilon (default: 1e-9)

    Returns:
        NoamScheduler wrapping the Adam optimizer
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0,  # Learning rate is controlled by the scheduler
        betas=betas,
        eps=eps,
    )
    return NoamScheduler(optimizer, d_model, warmup_steps, factor)
