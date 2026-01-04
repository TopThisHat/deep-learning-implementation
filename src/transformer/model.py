"""Complete Transformer model."""

import torch
import torch.nn as nn

from transformer.nn.encoder import Encoder
from transformer.nn.decoder import Decoder
from transformer.nn.generator import Generator
from transformer.functional import create_look_ahead_mask, create_padding_mask


class Transformer(nn.Module):
    """
    Full Transformer model from "Attention Is All You Need".

    Combines encoder, decoder, and generator into a complete
    sequence-to-sequence model.

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension (default: 512)
        n_layers: Number of encoder/decoder layers (default: 6)
        n_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward inner dimension (default: 2048)
        dropout: Dropout probability (default: 0.1)
        max_len: Maximum sequence length (default: 5000)
        pad_token: Padding token index (default: 0)
        share_embeddings: Whether to share embeddings between encoder and decoder (default: False)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_token: int = 0,
        share_embeddings: bool = False,
    ):
        super().__init__()

        self.pad_token = pad_token
        self.d_model = d_model

        # Encoder
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )

        # Decoder
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )

        # Generator (output projection)
        self.generator = Generator(d_model, tgt_vocab_size)

        # Optionally share embeddings
        if share_embeddings:
            assert src_vocab_size == tgt_vocab_size, \
                "Vocab sizes must match to share embeddings"
            self.decoder.embedding.weight = self.encoder.embedding.weight

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            tgt: Target token indices of shape (batch_size, tgt_seq_len)
            src_mask: Optional source padding mask
            tgt_mask: Optional target mask (combined look-ahead and padding)

        Returns:
            logits: Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src, self.pad_token)

        if tgt_mask is None:
            tgt_padding_mask = create_padding_mask(tgt, self.pad_token)
            tgt_look_ahead_mask = create_look_ahead_mask(tgt.size(1), device=tgt.device)
            tgt_mask = tgt_padding_mask * tgt_look_ahead_mask  # Element-wise multiply for float masks

        # Encode
        encoder_output, _ = self.encoder(src, src_mask)

        # Decode
        decoder_output, _, _ = self.decoder(tgt, encoder_output, tgt_mask, src_mask)

        # Generate logits
        logits = self.generator(decoder_output)

        return logits

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Encode source sequence.

        Args:
            src: Source token indices of shape (batch_size, src_seq_len)
            src_mask: Optional source padding mask

        Returns:
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            attention_weights: List of attention weights from each layer
        """
        if src_mask is None:
            src_mask = create_padding_mask(src, self.pad_token)
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Decode target sequence given encoder output.

        Args:
            tgt: Target token indices of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional source padding mask for cross-attention
            tgt_mask: Optional target mask (combined look-ahead and padding)

        Returns:
            decoder_output: Decoder output of shape (batch_size, tgt_seq_len, d_model)
            self_attention_weights: List of self-attention weights
            cross_attention_weights: List of cross-attention weights
        """
        if tgt_mask is None:
            tgt_padding_mask = create_padding_mask(tgt, self.pad_token)
            tgt_look_ahead_mask = create_look_ahead_mask(tgt.size(1), device=tgt.device)
            tgt_mask = tgt_padding_mask * tgt_look_ahead_mask  # Element-wise multiply for float masks

        return self.decoder(tgt, encoder_output, tgt_mask, src_mask)
