"""
Train a Transformer from scratch on a copy task.

This example demonstrates training the "Attention Is All You Need" Transformer
on a simple sequence copying task, where the model learns to copy input sequences
to the output.

Usage:
    python examples/train_copy_task.py

The copy task is a common benchmark for sequence-to-sequence models because:
1. It's simple to understand and verify
2. It requires the model to learn positional relationships
3. It tests the full encoder-decoder architecture
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import our transformer library
import sys
sys.path.insert(0, "src")

from transformer import Transformer, LabelSmoothingLoss, create_noam_optimizer
from transformer.functional import create_look_ahead_mask, create_padding_mask


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    # Model architecture
    vocab_size: int = 12          # 0=pad, 1=sos, 2=eos, 3-11=tokens
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 50
    
    # Training
    batch_size: int = 64
    num_epochs: int = 20
    warmup_steps: int = 400
    label_smoothing: float = 0.1
    
    # Data
    seq_len: int = 10             # Length of sequences to copy
    train_samples: int = 10000
    val_samples: int = 1000
    
    # Special tokens
    pad_token: int = 0
    sos_token: int = 1
    eos_token: int = 2
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# =============================================================================
# Dataset
# =============================================================================

class CopyDataset(Dataset):
    """
    Dataset for the copy task.
    
    Given an input sequence like [1, 5, 3, 7, 2], the target is to output
    the same sequence: [1, 5, 3, 7, 2].
    
    Format:
        src: [SOS, tokens..., EOS, PAD...]
        tgt_input: [SOS, tokens..., PAD...]
        tgt_output: [tokens..., EOS, PAD...]
    """
    
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, config: TrainConfig):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.config = config
        
        # Pre-generate all data
        self.data = self._generate_data()
    
    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # Generate random tokens (excluding special tokens 0, 1, 2)
            tokens = torch.randint(3, self.vocab_size, (self.seq_len,))
            
            # Source: [SOS, tokens..., EOS]
            src = torch.cat([
                torch.tensor([self.config.sos_token]),
                tokens,
                torch.tensor([self.config.eos_token]),
            ])
            
            # Target input (for teacher forcing): [SOS, tokens...]
            tgt_input = torch.cat([
                torch.tensor([self.config.sos_token]),
                tokens,
            ])
            
            # Target output (labels): [tokens..., EOS]
            tgt_output = torch.cat([
                tokens,
                torch.tensor([self.config.eos_token]),
            ])
            
            data.append((src, tgt_input, tgt_output))
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Collate function to pad sequences in a batch."""
    srcs, tgt_inputs, tgt_outputs = zip(*batch)
    
    # Stack (all sequences have same length in this simple example)
    src_batch = torch.stack(srcs)
    tgt_input_batch = torch.stack(tgt_inputs)
    tgt_output_batch = torch.stack(tgt_outputs)
    
    return src_batch, tgt_input_batch, tgt_output_batch


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    criterion: nn.Module,
    config: TrainConfig,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for src, tgt_input, tgt_output in dataloader:
        src = src.to(config.device)
        tgt_input = tgt_input.to(config.device)
        tgt_output = tgt_output.to(config.device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(src, tgt_input)
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.view(-1, config.vocab_size)
        targets = tgt_output.view(-1)
        
        loss = criterion(log_probs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track loss (count non-padding tokens)
        n_tokens = (tgt_output != config.pad_token).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens
    
    return total_loss / total_tokens


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: TrainConfig,
) -> tuple[float, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct = 0
    total = 0
    
    for src, tgt_input, tgt_output in dataloader:
        src = src.to(config.device)
        tgt_input = tgt_input.to(config.device)
        tgt_output = tgt_output.to(config.device)
        
        # Forward pass
        logits = model(src, tgt_input)
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_flat = log_probs.view(-1, config.vocab_size)
        targets_flat = tgt_output.view(-1)
        
        loss = criterion(log_probs_flat, targets_flat)
        
        # Track loss
        n_tokens = (tgt_output != config.pad_token).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens
        
        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        mask = tgt_output != config.pad_token
        correct += ((predictions == tgt_output) & mask).sum().item()
        total += mask.sum().item()
    
    avg_loss = total_loss / total_tokens
    accuracy = correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    max_len: int,
    config: TrainConfig,
) -> torch.Tensor:
    """Greedy decoding for inference."""
    model.eval()
    
    # Encode source
    encoder_output, _ = model.encode(src)
    
    # Start with SOS token
    batch_size = src.size(0)
    decoded = torch.full((batch_size, 1), config.sos_token, dtype=torch.long, device=config.device)
    
    for _ in range(max_len - 1):
        # Create masks
        tgt_mask = create_look_ahead_mask(decoded.size(1), device=config.device)
        
        # Decode
        decoder_output, _, _ = model.decode(decoded, encoder_output, tgt_mask=tgt_mask)
        
        # Get next token
        logits = model.generator(decoder_output[:, -1:, :])
        next_token = logits.argmax(dim=-1)
        
        # Append to decoded sequence
        decoded = torch.cat([decoded, next_token], dim=1)
        
        # Stop if all sequences have produced EOS
        if (next_token == config.eos_token).all():
            break
    
    return decoded


def demo_inference(model: nn.Module, config: TrainConfig):
    """Demonstrate model inference on a few examples."""
    print("\n" + "=" * 60)
    print("Inference Demo")
    print("=" * 60)
    
    model.eval()
    
    # Generate a few test sequences
    for i in range(3):
        # Random sequence
        tokens = torch.randint(3, config.vocab_size, (config.seq_len,))
        src = torch.cat([
            torch.tensor([config.sos_token]),
            tokens,
            torch.tensor([config.eos_token]),
        ]).unsqueeze(0).to(config.device)
        
        # Decode
        decoded = greedy_decode(model, src, max_len=config.seq_len + 2, config=config)
        
        # Print results
        src_tokens = src[0, 1:-1].tolist()  # Remove SOS and EOS
        pred_tokens = decoded[0, 1:].tolist()  # Remove SOS
        
        # Remove EOS and everything after from prediction
        if config.eos_token in pred_tokens:
            pred_tokens = pred_tokens[:pred_tokens.index(config.eos_token)]
        
        match = "✓" if src_tokens == pred_tokens else "✗"
        print(f"\nExample {i + 1}:")
        print(f"  Input:  {src_tokens}")
        print(f"  Output: {pred_tokens}")
        print(f"  Match:  {match}")


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    print("=" * 60)
    print("Training Transformer on Copy Task")
    print("=" * 60)
    
    # Configuration
    config = TrainConfig()
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Model: d_model={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")
    print(f"  Training: {config.num_epochs} epochs, batch_size={config.batch_size}")
    print(f"  Task: Copy sequences of length {config.seq_len}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CopyDataset(config.train_samples, config.seq_len, config.vocab_size, config)
    val_dataset = CopyDataset(config.val_samples, config.seq_len, config.vocab_size, config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Create model
    print("\nCreating model...")
    model = Transformer(
        src_vocab_size=config.vocab_size,
        tgt_vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
        pad_token=config.pad_token,
    ).to(config.device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    
    # Create optimizer with Noam schedule
    optimizer = create_noam_optimizer(
        model,
        d_model=config.d_model,
        warmup_steps=config.warmup_steps,
    )
    
    # Create loss function with label smoothing
    criterion = LabelSmoothingLoss(
        vocab_size=config.vocab_size,
        smoothing=config.label_smoothing,
        pad_token=config.pad_token,
    )
    
    # Training loop
    print("\nTraining...")
    print("-" * 60)
    
    best_val_acc = 0
    
    for epoch in range(1, config.num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, config)
        
        elapsed = time.time() - start_time
        lr = optimizer.get_lr()
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            marker = " *"
        else:
            marker = ""
        
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed:.1f}s{marker}"
        )
        
        # Early stopping if perfect accuracy
        if val_acc >= 0.99:
            print(f"\nReached {val_acc:.2%} accuracy, stopping early!")
            break
    
    print("-" * 60)
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    
    # Demo inference
    demo_inference(model, config)
    
    # Save model
    save_path = "transformer_copy_task.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
