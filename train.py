#!/usr/bin/env python3
"""Training loop for the Transformer LM on TinyStories."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from eecs148b_hw1.model import TransformerLM
from eecs148b_hw1.loss import cross_entropy_loss
from eecs148b_hw1.data import get_batch
from eecs148b_hw1.bpe import Tokenizer
from eecs148b_hw1.decode import generate


def cosine_lr_schedule(step: int, warmup_steps: int, total_steps: int, lr: float, min_lr: float = 0.0) -> float:
    """Cosine learning rate with linear warmup."""
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def evaluate(model, val_data, batch_size, context_length, device, num_batches=20):
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(val_data, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Transformer LM on TinyStories")
    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Path to tokenized training data (.npy)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to tokenized validation data (.npy)")
    parser.add_argument("--vocab_path", type=str, default=None, help="Path to vocab JSON for generation")
    parser.add_argument("--merges_path", type=str, default=None, help="Path to merges JSON for generation")
    # Model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--no_pos_emb", action="store_true", help="Disable positional embeddings (NoPE)")
    parser.add_argument("--no_layer_norm", action="store_true", help="Disable layer normalization")
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--total_steps", type=int, default=2500)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load data (memory-mapped)
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")
    print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

    device = args.device

    # Build model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_pos_emb=not args.no_pos_emb,
        use_layer_norm=not args.no_layer_norm,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer (AdamW with decoupled weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # Logging
    log = {
        "config": vars(args),
        "num_params": num_params,
        "steps": [],
    }

    best_val_loss = float("inf")
    model.train()
    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        # LR schedule
        lr = cosine_lr_schedule(step, args.warmup_steps, args.total_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            entry = {
                "step": step,
                "train_loss": loss.item(),
                "lr": lr,
                "wallclock": elapsed,
            }
            log["steps"].append(entry)
            print(f"step {step:>5d} | train_loss {loss.item():.4f} | lr {lr:.2e} | time {elapsed:.1f}s")

        if step % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device)
            perplexity = math.exp(val_loss)
            print(f"           val_loss {val_loss:.4f} | ppl {perplexity:.2f}")
            if log["steps"]:
                log["steps"][-1]["val_loss"] = val_loss
                log["steps"][-1]["perplexity"] = perplexity

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"           saved best checkpoint → {ckpt_path}")

    # Save final checkpoint and log
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "final.pt"))
    with open(os.path.join(args.checkpoint_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("Training complete.")


if __name__ == "__main__":
    main()
