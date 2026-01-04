"""Loss functions for Transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss as described in "Attention Is All You Need".

    Instead of using hard targets (one-hot), this distributes some probability
    mass to other classes, which helps prevent overconfidence and improves
    generalization.

    For a target class, the smoothed target becomes:
        - Target class: 1 - smoothing
        - Other classes: smoothing / (vocab_size - 1)
        - Padding positions are ignored

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

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw logits of shape (batch_size * seq_len, vocab_size)
            target: Target token indices of shape (batch_size * seq_len,)

        Returns:
            loss: Scalar loss value (mean over non-padding tokens)
        """
        # Apply log softmax to get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create mask for non-padding positions
        non_pad_mask = target != self.pad_token
        
        # Count non-padding tokens for normalization
        n_tokens = non_pad_mask.sum().item()
        
        if n_tokens == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Get log probs for the target tokens (negative log likelihood part)
        # Gather the log probs at target indices
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        
        # Smoothing loss: negative mean of all log probs (entropy-like term)
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Combine: (1 - smoothing) * NLL + smoothing * smooth_loss
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        # Apply mask and compute mean over non-padding tokens
        loss = loss.masked_select(non_pad_mask).sum() / n_tokens
        
        return loss
