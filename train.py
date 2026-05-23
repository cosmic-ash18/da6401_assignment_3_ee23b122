"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT:
  greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device)
      -> torch.Tensor  shape [1, out_len]
  evaluate_bleu(model, test_dataloader, tgt_vocab, device) -> float 0-100
  save_checkpoint(model, optimizer, scheduler, epoch, path) -> None
  load_checkpoint(path, model, optimizer, scheduler) -> int
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Multi30kDataset, PAD_IDX, SOS_IDX, EOS_IDX
from lr_scheduler import NoamScheduler
import model as model_module
from model import Transformer, make_src_mask, make_tgt_mask


# =============================================================================
# Label smoothing loss
# =============================================================================

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for token classification."""

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in [0, 1)")
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch*tgt_len, vocab_size]
            target: [batch*tgt_len]
        """
        log_probs = F_log_softmax(logits, dim=-1)
        non_pad = target != self.pad_idx

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            if self.vocab_size <= 2:
                true_dist.fill_(0.0)
            else:
                true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist[:, self.pad_idx] = 0.0
            safe_target = target.clone()
            safe_target[~non_pad] = self.pad_idx
            true_dist.scatter_(1, safe_target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0
            true_dist[~non_pad] = 0.0

        denom = non_pad.sum().clamp_min(1)
        return torch.sum(-true_dist * log_probs) / denom


# Local alias keeps the class easy to unit-test/mocking-friendly.
def F_log_softmax(x, dim=-1):
    return torch.nn.functional.log_softmax(x, dim=dim)


def _current_lr(optimizer: Optional[torch.optim.Optimizer]) -> float:
    if optimizer is None or not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _mha_qk_grad_norms(model: Transformer) -> dict[str, float]:
    """Average gradient norms for Query/Key projection weights across all MHA blocks."""
    q_norms = []
    k_norms = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name.endswith("W_q.weight"):
            q_norms.append(float(param.grad.detach().norm().item()))
        elif name.endswith("W_k.weight"):
            k_norms.append(float(param.grad.detach().norm().item()))
    out = {}
    if q_norms:
        out["grad_norm/query_weights"] = sum(q_norms) / len(q_norms)
    if k_norms:
        out["grad_norm/key_weights"] = sum(k_norms) / len(k_norms)
    return out


def _correct_token_confidence(logits: torch.Tensor, target: torch.Tensor, pad_idx: int) -> float:
    """Mean p_model(correct_token) over non-pad target positions."""
    with torch.no_grad():
        mask = target != pad_idx
        if int(mask.sum().item()) == 0:
            return 0.0
        probs = torch.softmax(logits.detach(), dim=-1)
        conf = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return float(conf[mask].mean().item())


# =============================================================================
# Training / evaluation epoch
# =============================================================================

def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        return batch["src"], batch["tgt"]
    raise TypeError("Each batch must be (src, tgt) or {'src': ..., 'tgt': ...}")


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    wandb_run=None,
    step_state: Optional[dict] = None,
    log_every_steps: int = 50,
    log_grad_steps: int = 0,
    log_prediction_confidence: bool = False,
) -> float:
    """Run one epoch of training or evaluation."""
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0

    iterator = tqdm(data_iter, desc=f"epoch {epoch_num} {'train' if is_train else 'eval'}", leave=False)
    for batch in iterator:
        src, tgt = _unpack_batch(batch)
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_gold = tgt[:, 1:]

        src_mask = make_src_mask(src, getattr(model, "src_pad_idx", PAD_IDX))
        tgt_mask = make_tgt_mask(tgt_input, getattr(model, "tgt_pad_idx", PAD_IDX))

        if is_train:
            assert optimizer is not None, "optimizer must be provided during training"
            optimizer.zero_grad(set_to_none=True)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_gold.reshape(-1))

        global_step = None
        if is_train and step_state is not None:
            step_state["step"] = int(step_state.get("step", 0)) + 1
            global_step = int(step_state["step"])

        grad_metrics = {}
        confidence = None
        if is_train:
            loss.backward()
            if wandb_run is not None and global_step is not None and global_step <= log_grad_steps:
                grad_metrics = _mha_qk_grad_norms(model)
            if log_prediction_confidence:
                confidence = _correct_token_confidence(logits, tgt_gold, getattr(model, "tgt_pad_idx", PAD_IDX))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if wandb_run is not None and global_step is not None:
                should_log = (global_step % max(1, log_every_steps) == 0) or bool(grad_metrics)
                if should_log:
                    metrics = {
                        "step/train_batch_loss": float(loss.detach().item()),
                        "step/lr": _current_lr(optimizer),
                    }
                    metrics.update(grad_metrics)
                    if confidence is not None:
                        metrics["step/correct_token_confidence"] = confidence
                    wandb_run.log(metrics, step=global_step)

        total_loss += float(loss.detach().item())
        total_batches += 1
        iterator.set_postfix(loss=f"{total_loss / max(1, total_batches):.4f}")

    return total_loss / max(1, total_batches)


@torch.no_grad()
def evaluate_token_accuracy(data_iter, model: Transformer, device: str = "cpu") -> float:
    """Teacher-forced non-pad token accuracy, useful for W&B validation curves."""
    model.eval()
    correct = 0
    total = 0
    for batch in data_iter:
        src, tgt = _unpack_batch(batch)
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_gold = tgt[:, 1:]
        src_mask = make_src_mask(src, getattr(model, "src_pad_idx", PAD_IDX))
        tgt_mask = make_tgt_mask(tgt_input, getattr(model, "tgt_pad_idx", PAD_IDX))
        logits = model(src, tgt_input, src_mask, tgt_mask)
        pred = logits.argmax(dim=-1)
        mask = tgt_gold != getattr(model, "tgt_pad_idx", PAD_IDX)
        correct += int(((pred == tgt_gold) & mask).sum().item())
        total += int(mask.sum().item())
    return correct / max(1, total)


# =============================================================================
# Greedy decoding
# =============================================================================

@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a translation token-by-token using greedy decoding."""
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)

    ys = torch.full((src.size(0), 1), start_symbol, dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, getattr(model, "tgt_pad_idx", PAD_IDX))
        logits = model.decode(memory, src_mask, ys, tgt_mask)
        next_word = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ys = torch.cat([ys, next_word], dim=1)
        if src.size(0) == 1 and int(next_word.item()) == end_symbol:
            break
    return ys


# =============================================================================
# BLEU evaluation
# =============================================================================

def _token_from_vocab(vocab, idx: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(int(idx))
    if hasattr(vocab, "itos"):
        return vocab.itos[int(idx)]
    raise TypeError("tgt_vocab must provide lookup_token(idx) or itos[idx]")


def _ids_to_tokens(ids, vocab) -> list[str]:
    tokens = []
    for idx in ids:
        tok = _token_from_vocab(vocab, int(idx))
        if tok in {"<pad>", "<sos>"}:
            continue
        if tok == "<eos>":
            break
        tokens.append(tok)
    return tokens


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def _corpus_bleu(predictions: list[list[str]], references: list[list[str]], max_n: int = 4) -> float:
    """Small corpus BLEU implementation returning 0-100."""
    pred_len = sum(len(p) for p in predictions)
    ref_len = sum(len(r) for r in references)
    if pred_len == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        clipped = 0
        total = 0
        for pred, ref in zip(predictions, references):
            pred_counts = _ngram_counts(pred, n)
            ref_counts = _ngram_counts(ref, n)
            total += sum(pred_counts.values())
            for gram, count in pred_counts.items():
                clipped += min(count, ref_counts.get(gram, 0))
        # Add-1 smoothing avoids BLEU becoming exactly zero for early weak models.
        precisions.append((clipped + 1.0) / (total + 1.0))

    log_precision = sum(math.log(p) for p in precisions) / max_n
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(1, pred_len))
    return 100.0 * brevity_penalty * math.exp(log_precision)


@torch.no_grad()
def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """Evaluate translation quality with corpus-level BLEU score."""
    model.eval()
    predictions: list[list[str]] = []
    references: list[list[str]] = []

    start_symbol = getattr(model, "tgt_sos_idx", SOS_IDX)
    end_symbol = getattr(model, "tgt_eos_idx", EOS_IDX)
    src_pad_idx = getattr(model, "src_pad_idx", PAD_IDX)

    for batch in tqdm(test_dataloader, desc="BLEU", leave=False):
        src_batch, tgt_batch = _unpack_batch(batch)
        for i in range(src_batch.size(0)):
            src = src_batch[i : i + 1].to(device)
            tgt = tgt_batch[i].tolist()
            src_mask = make_src_mask(src, src_pad_idx)
            pred_ids = greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device=device)
            predictions.append(_ids_to_tokens(pred_ids.squeeze(0).tolist(), tgt_vocab))
            references.append(_ids_to_tokens(tgt, tgt_vocab))

    return _corpus_bleu(predictions, references)


@torch.no_grad()
def log_encoder_attention_maps(
    model: Transformer,
    data_loader: DataLoader,
    src_vocab,
    device: str,
    wandb_run,
    max_heads: int | None = None,
) -> None:
    """Log heatmaps for the last encoder-layer self-attention heads for one validation example."""
    if wandb_run is None:
        return

    model.eval()
    batch = next(iter(data_loader))
    src_batch, tgt_batch = _unpack_batch(batch)
    src = src_batch[:1].to(device)
    tgt = tgt_batch[:1].to(device)
    tgt_input = tgt[:, :-1]

    src_mask = make_src_mask(src, getattr(model, "src_pad_idx", PAD_IDX))
    tgt_mask = make_tgt_mask(tgt_input, getattr(model, "tgt_pad_idx", PAD_IDX))
    _ = model(src, tgt_input, src_mask, tgt_mask)

    attn = model.encoder.layers[-1].self_attn.attn_weights
    if attn is None:
        return
    attn = attn[0].detach().cpu()  # [heads, src_len, src_len]
    src_ids = src.squeeze(0).detach().cpu().tolist()
    tokens = [_token_from_vocab(src_vocab, i) for i in src_ids]
    if max_heads is not None:
        attn = attn[:max_heads]

    log_payload = {"attention/sample_source_tokens": " ".join(tokens)}
    for h in range(attn.size(0)):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(attn[h].numpy(), aspect="auto")
        ax.set_title(f"Last encoder layer - head {h}")
        ax.set_xlabel("Key/source token attended to")
        ax.set_ylabel("Query/source token")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        import wandb as _wandb
        log_payload[f"attention/last_encoder_head_{h}"] = _wandb.Image(fig)
        plt.close(fig)

    wandb_run.log(log_payload)


# =============================================================================
# Checkpoint utilities
# =============================================================================

def _serialise_vocab(vocab):
    if vocab is None:
        return None
    if hasattr(vocab, "to_serializable"):
        return vocab.to_serializable()
    if hasattr(vocab, "stoi") and hasattr(vocab, "itos"):
        return {"stoi": dict(vocab.stoi), "itos": list(vocab.itos)}
    return None


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """Save model + optimizer + scheduler state to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": model.get_config() if hasattr(model, "get_config") else {
            "src_vocab_size": model.src_vocab_size,
            "tgt_vocab_size": model.tgt_vocab_size,
            "d_model": model.d_model,
            "N": model.N,
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "dropout": model.dropout_p,
        },
        "src_vocab": _serialise_vocab(getattr(model, "src_vocab", None)),
        "tgt_vocab": _serialise_vocab(getattr(model, "tgt_vocab", None)),
    }
    torch.save(payload, path)


def _safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """Restore model and optionally optimizer/scheduler state from disk."""
    checkpoint = _safe_torch_load(path, map_location=next(model.parameters()).device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)

    if hasattr(model, "_load_vocab_metadata") and isinstance(checkpoint, dict):
        model._load_vocab_metadata(checkpoint)

    if optimizer is not None and isinstance(checkpoint, dict) and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and isinstance(checkpoint, dict) and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0


# =============================================================================
# Experiment entry point
# =============================================================================

def run_training_experiment() -> None:
    parser = argparse.ArgumentParser(description="Train a Transformer on Multi30k German->English")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--max_vocab_size", type=int, default=12000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="best_checkpoint.pth")
    parser.add_argument("--last_checkpoint", type=str, default="last_checkpoint.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fixed_lr", action="store_true", help="Use constant lr=1e-4 instead of Noam")
    parser.add_argument("--lr", type=float, default=1.0, help="Base lr. Use 1.0 with Noam, 1e-4 with --fixed_lr")
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="da6401-a3")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--experiment_tag", type=str, default="report")
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--log_grad_steps", type=int, default=0)
    parser.add_argument("--log_prediction_confidence", action="store_true")
    parser.add_argument("--log_attention_sample", action="store_true")
    parser.add_argument("--compute_val_bleu", action="store_true", help="Compute final validation BLEU after loading best checkpoint")
    parser.add_argument("--no_scale_attention", action="store_true", help="Report ablation: remove the 1/sqrt(d_k) attention scale")
    parser.add_argument("--learned_positional", action="store_true", help="Report ablation: use learned positional embeddings instead of sinusoidal PE")
    args = parser.parse_args()

    if args.fixed_lr and args.lr == 1.0:
        args.lr = 1e-4

    # Default is the paper-correct scaled attention. Only disable it for the W&B ablation run.
    model_module.USE_ATTENTION_SCALE = not args.no_scale_attention

    device = torch.device(args.device)

    wandb_run = None
    if args.use_wandb:
        import wandb as _wandb
        tags = [args.experiment_tag]
        if args.no_scale_attention:
            tags.append("no-scale-attention")
        if args.fixed_lr:
            tags.append("fixed-lr")
        if args.learned_positional:
            tags.append("learned-positional")
        if args.smoothing == 0.0:
            tags.append("no-label-smoothing")
        wandb_run = _wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
            tags=tags,
        )

    print("Loading Multi30k and building vocab...")
    train_ds = Multi30kDataset(
        split="train",
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        max_len=args.max_len,
    )
    src_vocab, tgt_vocab = train_ds.build_vocab()
    train_ds.process_data()

    val_ds = Multi30kDataset(
        split="validation",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        max_len=args.max_len,
    )
    test_ds = Multi30kDataset(
        split="test",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        max_len=args.max_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_ds.collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_ds.collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=test_ds.collate_fn,
        num_workers=0,
    )

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        N=args.N,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        learned_positional=args.learned_positional,
        max_len=args.max_len,
    ).to(device)
    model.set_vocabs(src_vocab, tgt_vocab)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = None if args.fixed_lr else NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), pad_idx=PAD_IDX, smoothing=args.smoothing)

    best_val = float("inf")
    step_state = {"step": 0}
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler, epoch, True, str(device),
            wandb_run=wandb_run,
            step_state=step_state,
            log_every_steps=args.log_every_steps,
            log_grad_steps=args.log_grad_steps,
            log_prediction_confidence=args.log_prediction_confidence,
        )
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, False, str(device))
        val_acc = evaluate_token_accuracy(val_loader, model, str(device))
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_token_acc={val_acc:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch, args.last_checkpoint)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, args.checkpoint)
            print(f"Saved best checkpoint to {args.checkpoint}")

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "epoch/train_loss": train_loss, "epoch/val_loss": val_loss, "epoch/val_token_accuracy": val_acc, "epoch/best_val_loss": best_val}, step=step_state["step"])

    if args.log_attention_sample and wandb_run is not None:
        print("Logging last encoder-layer attention maps to W&B...")
        log_encoder_attention_maps(model, val_loader, src_vocab, str(device), wandb_run)

    print("Loading best checkpoint for BLEU evaluation...")
    load_checkpoint(args.checkpoint, model, optimizer=None, scheduler=None)
    if args.compute_val_bleu:
        val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=str(device), max_len=args.max_len)
        print(f"Validation BLEU: {val_bleu:.2f}")
        if wandb_run is not None:
            wandb_run.log({"val_bleu": val_bleu}, step=step_state["step"])
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=str(device), max_len=args.max_len)
    print(f"Test BLEU: {bleu:.2f}")
    if wandb_run is not None:
        wandb_run.log({"test_bleu": bleu}, step=step_state["step"])
        wandb_run.finish()


if __name__ == "__main__":
    run_training_experiment()
