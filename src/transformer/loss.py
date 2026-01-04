"""Loss functions for Transformer models."""

import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss as described in "Attention Is All You Need".

    Instead of using hard targets (one-hot), this distributes some probability
    mass to other classes, which helps prevent overconfidence and improves
    generalization.

    For a target class, the smoothed target becomes:
        - Target class: 1 - smoothing
        - Other classes: smoothing / (vocab_size - 2)
        - Padding token: 0

    Args:
        vocab_size: Size of the vocabulary
        smoothing: Label smoothing factor (default: 0.1)
        pad_token: Padding token index to ignore (default: 0)
    """

    def __init__(
        self,
        vocab_size: int,
        smoothing: float = 0.1,
        pad_token: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_token = pad_token
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(
        self,
        log_probs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            log_probs: Log probabilities of shape (batch_size * seq_len, vocab_size)
            target: Target token indices of shape (batch_size * seq_len,)

        Returns:
            loss: Scalar loss value
        """
        assert log_probs.size(1) == self.vocab_size

        # Create smoothed target distribution
        # Distribute smoothing mass to all classes except target and padding
        true_dist = log_probs.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))

        # Assign high confidence to ground truth labels
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # Zero out padding column (padding token should never receive probability)
        true_dist[:, self.pad_token] = 0

        # Zero out rows where target is padding (these positions don't contribute to loss)
        pad_mask = torch.nonzero(target.data == self.pad_token)
        if pad_mask.dim() > 0:
            true_dist.index_fill_(0, pad_mask.squeeze(), 0.0)

        return self.criterion(log_probs, true_dist.clone().detach())
