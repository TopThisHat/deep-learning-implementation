"""
Train a Transformer model on the Multi30k German-English translation task.

This example uses the HuggingFace datasets library to load the Multi30k dataset
and trains a BPE tokenizer for both source and target languages.

Usage:
    python examples/train_translation.py
"""

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, Dataset

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformer import Transformer
from transformer.functional import create_padding_mask
from transformer.loss import LabelSmoothingLoss
from transformer.optim import create_noam_optimizer


# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def train_tokenizer(texts: list[str], vocab_size: int = 8000) -> Tokenizer:
    """Train a BPE tokenizer on the given texts."""
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )
    
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


class TranslationDataset(Dataset):
    """Dataset for translation task."""
    
    def __init__(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_len: int = 128,
    ):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        
        # Get special token ids
        self.src_pad_id = src_tokenizer.token_to_id(PAD_TOKEN)
        self.src_bos_id = src_tokenizer.token_to_id(BOS_TOKEN)
        self.src_eos_id = src_tokenizer.token_to_id(EOS_TOKEN)
        
        self.tgt_pad_id = tgt_tokenizer.token_to_id(PAD_TOKEN)
        self.tgt_bos_id = tgt_tokenizer.token_to_id(BOS_TOKEN)
        self.tgt_eos_id = tgt_tokenizer.token_to_id(EOS_TOKEN)
    
    def __len__(self) -> int:
        return len(self.src_texts)
    
    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # Tokenize
        src_ids = self.src_tokenizer.encode(src_text).ids
        tgt_ids = self.tgt_tokenizer.encode(tgt_text).ids
        
        # Add BOS and EOS tokens, truncate if needed
        src_ids = [self.src_bos_id] + src_ids[: self.max_len - 2] + [self.src_eos_id]
        tgt_ids = [self.tgt_bos_id] + tgt_ids[: self.max_len - 2] + [self.tgt_eos_id]
        
        return src_ids, tgt_ids


def collate_fn(batch: list[tuple[list[int], list[int]]], src_pad_id: int, tgt_pad_id: int):
    """Collate function for DataLoader with dynamic padding."""
    src_batch, tgt_batch = zip(*batch)
    
    # Find max lengths in this batch
    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)
    
    # Pad sequences
    src_padded = []
    tgt_padded = []
    
    for src, tgt in zip(src_batch, tgt_batch):
        src_padded.append(src + [src_pad_id] * (src_max_len - len(src)))
        tgt_padded.append(tgt + [tgt_pad_id] * (tgt_max_len - len(tgt)))
    
    return torch.tensor(src_padded), torch.tensor(tgt_padded)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    scheduler,  # NoamScheduler (wraps optimizer)
    criterion: nn.Module,
    device: torch.device,
    src_pad_id: int,
    tgt_pad_id: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Target input (shift right) and target output
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Create padding masks
        src_mask = create_padding_mask(src, src_pad_id)
        tgt_mask = create_padding_mask(tgt_input, tgt_pad_id)
        
        # Forward pass
        scheduler.zero_grad()
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        # Compute loss (flatten for cross entropy)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scheduler.optimizer.step()
        scheduler.step()
        
        # Count non-padding tokens for accurate loss
        n_tokens = (tgt_output != tgt_pad_id).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / total_tokens
            lr = scheduler.get_lr()
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
    
    return total_loss / total_tokens


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    src_pad_id: int,
    tgt_pad_id: int,
) -> float:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        src_mask = create_padding_mask(src, src_pad_id)
        tgt_mask = create_padding_mask(tgt_input, tgt_pad_id)
        
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )
        
        n_tokens = (tgt_output != tgt_pad_id).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    
    return total_loss / total_tokens


@torch.no_grad()
def translate(
    model: nn.Module,
    src_text: str,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
    max_len: int = 64,
) -> str:
    """Translate a source sentence."""
    model.eval()
    
    # Get special token ids
    src_bos_id = src_tokenizer.token_to_id(BOS_TOKEN)
    src_eos_id = src_tokenizer.token_to_id(EOS_TOKEN)
    src_pad_id = src_tokenizer.token_to_id(PAD_TOKEN)
    
    tgt_bos_id = tgt_tokenizer.token_to_id(BOS_TOKEN)
    tgt_eos_id = tgt_tokenizer.token_to_id(EOS_TOKEN)
    tgt_pad_id = tgt_tokenizer.token_to_id(PAD_TOKEN)
    
    # Encode source
    src_ids = src_tokenizer.encode(src_text).ids
    src_ids = [src_bos_id] + src_ids + [src_eos_id]
    src = torch.tensor([src_ids], device=device)
    src_mask = create_padding_mask(src, src_pad_id)
    
    # Start with BOS token
    tgt_ids = [tgt_bos_id]
    
    for _ in range(max_len):
        tgt = torch.tensor([tgt_ids], device=device)
        tgt_mask = create_padding_mask(tgt, tgt_pad_id)
        
        logits = model(src, tgt, src_mask, tgt_mask)
        next_token = logits[0, -1].argmax().item()
        
        if next_token == tgt_eos_id:
            break
        
        tgt_ids.append(next_token)
    
    # Decode (skip BOS)
    return tgt_tokenizer.decode(tgt_ids[1:])


def main():
    # Hyperparameters
    d_model = 256
    n_heads = 8
    n_layers = 4
    d_ff = 1024
    dropout = 0.1
    max_len = 128
    vocab_size = 8000
    
    batch_size = 64
    n_epochs = 10
    warmup_steps = 2000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Multi30k dataset
    print("Loading Multi30k dataset...")
    try:
        dataset = load_dataset("bentrevett/multi30k")
    except Exception as e:
        print(f"Could not load bentrevett/multi30k: {e}")
        print("Trying alternative dataset source...")
        # Alternative: use WMT14 subset if Multi30k is unavailable
        dataset = load_dataset("wmt14", "de-en", split={"train": "train[:30000]", "validation": "validation[:1000]", "test": "test[:1000]"})
    
    # Extract texts
    if "translation" in dataset["train"].features:
        # WMT format
        train_src = [ex["translation"]["de"] for ex in dataset["train"]]
        train_tgt = [ex["translation"]["en"] for ex in dataset["train"]]
        val_src = [ex["translation"]["de"] for ex in dataset["validation"]]
        val_tgt = [ex["translation"]["en"] for ex in dataset["validation"]]
    else:
        # Multi30k format (bentrevett version)
        train_src = [ex["de"] for ex in dataset["train"]]
        train_tgt = [ex["en"] for ex in dataset["train"]]
        val_src = [ex["de"] for ex in dataset["validation"]]
        val_tgt = [ex["en"] for ex in dataset["validation"]]
    
    print(f"Train size: {len(train_src)}, Val size: {len(val_src)}")
    print(f"Sample: DE: {train_src[0]}")
    print(f"        EN: {train_tgt[0]}")
    
    # Train tokenizers
    print("\nTraining tokenizers...")
    src_tokenizer = train_tokenizer(train_src, vocab_size=vocab_size)
    tgt_tokenizer = train_tokenizer(train_tgt, vocab_size=vocab_size)
    
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")
    
    src_pad_id = src_tokenizer.token_to_id(PAD_TOKEN)
    tgt_pad_id = tgt_tokenizer.token_to_id(PAD_TOKEN)
    
    # Create datasets
    train_dataset = TranslationDataset(
        train_src, train_tgt, src_tokenizer, tgt_tokenizer, max_len=max_len
    )
    val_dataset = TranslationDataset(
        val_src, val_tgt, src_tokenizer, tgt_tokenizer, max_len=max_len
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_pad_id, tgt_pad_id),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, src_pad_id, tgt_pad_id),
        num_workers=0,
    )
    
    # Create model
    print("\nCreating model...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len + 10,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create optimizer and scheduler (NoamScheduler wraps optimizer)
    scheduler = create_noam_optimizer(
        model, d_model=d_model, warmup_steps=warmup_steps, factor=1.0
    )
    
    # Loss function
    criterion = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_token=tgt_pad_id,
        smoothing=0.1,
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(
            model, train_loader, scheduler, criterion,
            device, src_pad_id, tgt_pad_id
        )
        val_loss = evaluate(model, val_loader, criterion, device, src_pad_id, tgt_pad_id)
        
        elapsed = time.time() - start_time
        
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):.2f}")
        print(f"  Val Loss: {val_loss:.4f} | Val PPL: {math.exp(val_loss):.2f}")
        print(f"  Time: {elapsed:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "src_vocab_size": src_vocab_size,
                "tgt_vocab_size": tgt_vocab_size,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
            }, "best_translation_model.pt")
            print("  ✓ Saved best model")
        
        # Sample translations
        print("\n  Sample translations:")
        for src_text in val_src[:3]:
            translation = translate(model, src_text, src_tokenizer, tgt_tokenizer, device)
            print(f"    DE: {src_text}")
            print(f"    EN: {translation}")
            print()
    
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
