"""Tests for transformer library."""

import torch
import pytest


class TestScaledDotProductAttention:
    """Tests for scaled_dot_product_attention function."""

    def test_output_shape(self):
        from transformer.functional import scaled_dot_product_attention

        batch_size, n_heads, seq_len, d_k = 2, 8, 10, 64
        query = torch.randn(batch_size, n_heads, seq_len, d_k)
        key = torch.randn(batch_size, n_heads, seq_len, d_k)
        value = torch.randn(batch_size, n_heads, seq_len, d_k)

        output, weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch_size, n_heads, seq_len, d_k)
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        from transformer.functional import scaled_dot_product_attention

        query = torch.randn(2, 8, 10, 64)
        key = torch.randn(2, 8, 10, 64)
        value = torch.randn(2, 8, 10, 64)

        _, weights = scaled_dot_product_attention(query, key, value)

        # Weights should sum to 1 along the last dimension
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""

    def test_output_shape(self):
        from transformer.nn import MultiHeadAttention

        d_model, n_heads = 512, 8
        batch_size, seq_len = 2, 10

        attention = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output, weights = attention(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_d_model_divisibility(self):
        from transformer.nn import MultiHeadAttention

        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=512, n_heads=7)  # 512 not divisible by 7


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""

    def test_output_shape(self):
        from transformer.nn import PositionalEncoding

        d_model = 512
        batch_size, seq_len = 2, 100

        pe = PositionalEncoding(d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        output = pe(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_adds_positional_info(self):
        from transformer.nn import PositionalEncoding

        d_model = 512
        pe = PositionalEncoding(d_model, dropout=0.0)
        
        x = torch.zeros(1, 10, d_model)
        output = pe(x)

        # Output should not be all zeros (positional info added)
        assert not torch.allclose(output, torch.zeros_like(output))


class TestTransformer:
    """Tests for complete Transformer model."""

    def test_forward_pass(self):
        from transformer import Transformer

        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
        )

        batch_size, src_len, tgt_len = 2, 10, 8
        src = torch.randint(1, 1000, (batch_size, src_len))
        tgt = torch.randint(1, 1000, (batch_size, tgt_len))

        logits = model(src, tgt)

        assert logits.shape == (batch_size, tgt_len, 1000)

    def test_encode_decode_separately(self):
        from transformer import Transformer

        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
        )

        src = torch.randint(1, 1000, (2, 10))
        tgt = torch.randint(1, 1000, (2, 8))

        # Encode
        encoder_output, enc_attn = model.encode(src)
        assert encoder_output.shape == (2, 10, 64)

        # Decode
        decoder_output, self_attn, cross_attn = model.decode(tgt, encoder_output)
        assert decoder_output.shape == (2, 8, 64)


class TestMasks:
    """Tests for mask creation functions."""

    def test_look_ahead_mask(self):
        from transformer.functional import create_look_ahead_mask

        mask = create_look_ahead_mask(5)

        assert mask.shape == (1, 1, 5, 5)
        # Lower triangular
        assert mask[0, 0, 0, 1] == 0  # Can't look ahead
        assert mask[0, 0, 1, 0] == 1  # Can look back
        assert mask[0, 0, 2, 2] == 1  # Can look at self

    def test_padding_mask(self):
        from transformer.functional import create_padding_mask

        seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        mask = create_padding_mask(seq, pad_token=0)

        assert mask.shape == (2, 1, 1, 5)
        assert mask[0, 0, 0, 2] == 1  # Non-padding
        assert mask[0, 0, 0, 3] == 0  # Padding
        assert mask[1, 0, 0, 1] == 1  # Non-padding
        assert mask[1, 0, 0, 2] == 0  # Padding


class TestNoamScheduler:
    """Tests for Noam learning rate scheduler."""

    def test_warmup_phase(self):
        from transformer.optim import NoamScheduler

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=100)

        # LR should increase during warmup
        lrs = []
        for _ in range(100):
            scheduler.step()
            lrs.append(scheduler.get_lr())

        # Check that LR is increasing
        assert lrs[-1] > lrs[0]

    def test_decay_phase(self):
        from transformer.optim import NoamScheduler

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=100)

        # Advance past warmup
        for _ in range(150):
            scheduler.step()

        lr_at_150 = scheduler.get_lr()

        for _ in range(50):
            scheduler.step()

        lr_at_200 = scheduler.get_lr()

        # LR should decrease after warmup
        assert lr_at_200 < lr_at_150
