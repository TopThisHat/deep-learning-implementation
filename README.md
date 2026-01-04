# Transformer

A clean, modular PyTorch implementation of the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import transformer

# Create a full Transformer model
model = transformer.Transformer(
    src_vocab_size=30000,
    tgt_vocab_size=30000,
    d_model=512,
    n_layers=6,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
)

# Forward pass
logits = model(src_tokens, tgt_tokens)
```

## Library Structure

The library is organized similar to PyTorch:

```
transformer/
├── __init__.py          # Main exports
├── model.py             # Complete Transformer model
├── functional.py        # Functional operations (attention, masks)
├── loss.py              # Loss functions (LabelSmoothingLoss)
├── optim.py             # Optimizers (NoamScheduler)
└── nn/                  # Neural network modules
    ├── attention.py     # MultiHeadAttention
    ├── embedding.py     # PositionalEncoding
    ├── feedforward.py   # FeedForward
    ├── encoder.py       # EncoderLayer, Encoder
    ├── decoder.py       # DecoderLayer, Decoder
    └── generator.py     # Generator
```

## Usage Examples

### Using Individual Components

```python
import torch
import transformer
import transformer.nn as nn

# Multi-Head Attention
attention = nn.MultiHeadAttention(d_model=512, n_heads=8)
output, weights = attention(query, key, value)

# Positional Encoding
pos_encoding = nn.PositionalEncoding(d_model=512)
x = pos_encoding(embeddings)

# Encoder
encoder = nn.Encoder(vocab_size=30000, d_model=512)
encoded, attn_weights = encoder(src_tokens)

# Decoder
decoder = nn.Decoder(vocab_size=30000, d_model=512)
decoded, self_attn, cross_attn = decoder(tgt_tokens, encoder_output)
```

### Functional API

```python
from transformer.functional import (
    scaled_dot_product_attention,
    create_look_ahead_mask,
    create_padding_mask,
)

# Compute attention
output, weights = scaled_dot_product_attention(query, key, value)

# Create masks
causal_mask = create_look_ahead_mask(seq_len, device=device)
padding_mask = create_padding_mask(tokens, pad_token=0)
```

### Training with Label Smoothing and Noam Optimizer

```python
from transformer import Transformer, LabelSmoothingLoss, create_noam_optimizer

# Model
model = Transformer(src_vocab_size=30000, tgt_vocab_size=30000, d_model=512)

# Loss with label smoothing
criterion = LabelSmoothingLoss(vocab_size=30000, smoothing=0.1)

# Optimizer with Noam schedule
optimizer = create_noam_optimizer(model, d_model=512, warmup_steps=4000)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    
    logits = model(batch.src, batch.tgt)
    log_probs = F.log_softmax(logits, dim=-1).view(-1, vocab_size)
    
    loss = criterion(log_probs, batch.target.view(-1))
    loss.backward()
    
    optimizer.step()
```

## Paper Reference

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## License

MIT
