"""
train.py
Train the Transformer model, saves checkpoints, and calculates BLEU.
"""

import argparse
import math
import os
from collections import Counter

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Multi30kDataset, PAD_IDX, SOS_IDX, EOS_IDX
from lr_scheduler import NoamScheduler
import model as model_module
from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    """Cross entropy style loss, but with label smoothing."""

    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()

        # Just a safety check so smoothing is not some invalid value.
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in [0, 1)")

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

        # If smoothing = 0.1, correct word gets 0.9 confidence.
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        # logits are raw model outputs.
        # target is the correct next word id.

        # Convert raw scores into log probabilities.
        log_probs = F_log_softmax(logits, dim=-1)

        # Padding is only for making batch sizes equal.
        # We should not punish the model for pad tokens.
        non_pad = target != self.pad_idx

        with torch.no_grad():
            # true_dist will store the target probability distribution.
            true_dist = torch.zeros_like(log_probs)

            # Give small probability to wrong tokens also.
            # This stops the model from becoming too confident.
            if self.vocab_size <= 2:
                true_dist.fill_(0.0)
            else:
                true_dist.fill_(self.smoothing / (self.vocab_size - 2))

            # Pad token should never be treated as a useful prediction.
            true_dist[:, self.pad_idx] = 0.0

            # For pad positions, scatter still needs a valid index.
            safe_target = target.clone()
            safe_target[~non_pad] = self.pad_idx

            # Put most of the probability on the correct word.
            true_dist.scatter_(1, safe_target.unsqueeze(1), self.confidence)

            # Remove all contribution from padding positions.
            true_dist[:, self.pad_idx] = 0.0
            true_dist[~non_pad] = 0.0

        # Divide only by the number of real tokens.
        denom = non_pad.sum().clamp_min(1)
        return torch.sum(-true_dist * log_probs) / denom


def F_log_softmax(x, dim=-1):
    # Small helper for log_softmax.
    return torch.nn.functional.log_softmax(x, dim=dim)


def _current_lr(optimizer):
    # Get the learning rate currently being used.
    if optimizer is None or not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _mha_qk_grad_norms(model):
    # Used only for the W&B scaling-factor experiment.
    # We check how large Query and Key gradients are.
    q_norms = []
    k_norms = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        # Query projection gradient.
        if name.endswith("W_q.weight"):
            q_norms.append(float(param.grad.detach().norm().item()))

        # Key projection gradient.
        elif name.endswith("W_k.weight"):
            k_norms.append(float(param.grad.detach().norm().item()))

    out = {}

    # Average over all attention layers.
    if q_norms:
        out["grad_norm/query_weights"] = sum(q_norms) / len(q_norms)

    if k_norms:
        out["grad_norm/key_weights"] = sum(k_norms) / len(k_norms)

    return out


def _correct_token_confidence(logits, target, pad_idx):
    # This measures how much probability the model gives to the correct word.
    # It is useful for the label smoothing experiment.
    with torch.no_grad():
        mask = target != pad_idx

        # If the whole thing is padding, return 0.
        if int(mask.sum().item()) == 0:
            return 0.0

        probs = torch.softmax(logits.detach(), dim=-1)

        # Pick probability of the correct class from the full vocab distribution.
        conf = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

        # Average only on real tokens.
        return float(conf[mask].mean().item())


def _unpack_batch(batch):
    # Most of our batches are simply (src, tgt).
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]

    # This is just extra support if a batch is given as a dict.
    if isinstance(batch, dict):
        return batch["src"], batch["tgt"]

    raise TypeError("Each batch must be (src, tgt) or {'src': ..., 'tgt': ...}")


def run_epoch(
    data_iter,
    model,
    loss_fn,
    optimizer,
    scheduler=None,
    epoch_num=0,
    is_train=True,
    device="cpu",
    wandb_run=None,
    step_state=None,
    log_every_steps=50,
    log_grad_steps=0,
    log_prediction_confidence=False,
):
    """Run one epoch. If is_train=True, it trains. Else, it only evaluates."""

    # This switches dropout etc. correctly for train/eval.
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0

    iterator = tqdm(
        data_iter,
        desc=f"epoch {epoch_num} {'train' if is_train else 'eval'}",
        leave=False,
    )

    for batch in iterator:
        src, tgt = _unpack_batch(batch)

        # Move tensors to CPU/GPU.
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing:
        # input to decoder is everything except last word.
        tgt_input = tgt[:, :-1]

        # expected output is everything except first word.
        tgt_gold = tgt[:, 1:]

        # Masks hide padding and future tokens.
        src_mask = make_src_mask(src, getattr(model, "src_pad_idx", PAD_IDX))
        tgt_mask = make_tgt_mask(tgt_input, getattr(model, "tgt_pad_idx", PAD_IDX))

        if is_train:
            assert optimizer is not None, "optimizer must be provided during training"

            # Clear old gradients before new backward pass.
            optimizer.zero_grad(set_to_none=True)

        # Forward pass through Transformer.
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Flatten batch and sequence dimensions for loss.
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            tgt_gold.reshape(-1),
        )

        global_step = None

        # Keep one global step counter for W&B.
        if is_train and step_state is not None:
            step_state["step"] = int(step_state.get("step", 0)) + 1
            global_step = int(step_state["step"])

        grad_metrics = {}
        confidence = None

        if is_train:
            # Backpropagation.
            loss.backward()

            # For first 1000 steps, we may log Q/K gradient norms.
            if wandb_run is not None and global_step is not None and global_step <= log_grad_steps:
                grad_metrics = _mha_qk_grad_norms(model)

            # For label smoothing experiment.
            if log_prediction_confidence:
                confidence = _correct_token_confidence(
                    logits,
                    tgt_gold,
                    getattr(model, "tgt_pad_idx", PAD_IDX),
                )

            # Prevent very large gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update model parameters.
            optimizer.step()

            # Update learning rate if scheduler is used.
            if scheduler is not None:
                scheduler.step()

            # Log training stats to W&B every few steps.
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

        # Show running average loss in tqdm.
        iterator.set_postfix(loss=f"{total_loss / max(1, total_batches):.4f}")

    # Return average loss for the epoch.
    return total_loss / max(1, total_batches)


@torch.no_grad()
def evaluate_token_accuracy(data_iter, model, device="cpu"):
    """Calculate validation accuracy at token level, ignoring pad tokens."""

    model.eval()

    correct = 0
    total = 0

    for batch in data_iter:
        src, tgt = _unpack_batch(batch)

        src = src.to(device)
        tgt = tgt.to(device)

        # Same teacher forcing split as training.
        tgt_input = tgt[:, :-1]
        tgt_gold = tgt[:, 1:]

        src_mask = make_src_mask(src, getattr(model, "src_pad_idx", PAD_IDX))
        tgt_mask = make_tgt_mask(tgt_input, getattr(model, "tgt_pad_idx", PAD_IDX))

        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Pick highest scoring token.
        pred = logits.argmax(dim=-1)

        # Only count real target words, not padding.
        mask = tgt_gold != getattr(model, "tgt_pad_idx", PAD_IDX)

        correct += int(((pred == tgt_gold) & mask).sum().item())
        total += int(mask.sum().item())

    return correct / max(1, total)


@torch.no_grad()
def greedy_decode(
    model,
    src,
    src_mask,
    max_len,
    start_symbol,
    end_symbol,
    device="cpu",
):
    """Translate by choosing the most likely next token each time."""

    model.eval()

    src = src.to(device)
    src_mask = src_mask.to(device)

    # Encode the German/source sentence once.
    memory = model.encode(src, src_mask)

    # Decoder starts with <sos>.
    ys = torch.full(
        (src.size(0), 1),
        start_symbol,
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_len - 1):
        # Mask so decoder cannot look ahead.
        tgt_mask = make_tgt_mask(ys, getattr(model, "tgt_pad_idx", PAD_IDX))

        # Decode all words generated so far.
        logits = model.decode(memory, src_mask, ys, tgt_mask)

        # Take the most likely next word.
        next_word = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        # Add this word to the generated sentence.
        ys = torch.cat([ys, next_word], dim=1)

        # Stop when <eos> comes for a single sentence.
        if src.size(0) == 1 and int(next_word.item()) == end_symbol:
            break

    return ys


def _token_from_vocab(vocab, idx):
    # Convert an ID back into a word.
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(int(idx))

    if hasattr(vocab, "itos"):
        return vocab.itos[int(idx)]

    raise TypeError("tgt_vocab must provide lookup_token(idx) or itos[idx]")


def _ids_to_tokens(ids, vocab):
    # Convert many IDs into words.
    tokens = []

    for idx in ids:
        tok = _token_from_vocab(vocab, int(idx))

        # These tokens are not real translation words.
        if tok in {"<pad>", "<sos>"}:
            continue

        # Stop after sentence ends.
        if tok == "<eos>":
            break

        tokens.append(tok)

    return tokens


def _ngram_counts(tokens, n):
    # Count word groups of length n.
    # Example for n=2: ["a", "man", "runs"] -> ("a","man"), ("man","runs")
    return Counter(
        tuple(tokens[i : i + n])
        for i in range(max(0, len(tokens) - n + 1))
    )


def _corpus_bleu(predictions, references, max_n=4):
    """Simple BLEU score implementation. Returns value from 0 to 100."""

    pred_len = sum(len(p) for p in predictions)
    ref_len = sum(len(r) for r in references)

    # If model predicts nothing, BLEU is zero.
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

            # Count only n-grams that also appear in reference.
            for gram, count in pred_counts.items():
                clipped += min(count, ref_counts.get(gram, 0))

        # Add-1 smoothing avoids zero BLEU when model is weak.
        precisions.append((clipped + 1.0) / (total + 1.0))

    # BLEU uses geometric mean of n-gram precisions.
    log_precision = sum(math.log(p) for p in precisions) / max_n

    # Short predictions should be penalized.
    if pred_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - ref_len / max(1, pred_len))

    return 100.0 * brevity_penalty * math.exp(log_precision)


@torch.no_grad()
def evaluate_bleu(
    model,
    test_dataloader,
    tgt_vocab,
    device="cpu",
    max_len=100,
):
    """Run greedy decoding and calculate BLEU score."""

    model.eval()

    predictions = []
    references = []

    start_symbol = getattr(model, "tgt_sos_idx", SOS_IDX)
    end_symbol = getattr(model, "tgt_eos_idx", EOS_IDX)
    src_pad_idx = getattr(model, "src_pad_idx", PAD_IDX)

    for batch in tqdm(test_dataloader, desc="BLEU", leave=False):
        src_batch, tgt_batch = _unpack_batch(batch)

        # Decode one sentence at a time.
        for i in range(src_batch.size(0)):
            src = src_batch[i : i + 1].to(device)
            tgt = tgt_batch[i].tolist()

            src_mask = make_src_mask(src, src_pad_idx)

            pred_ids = greedy_decode(
                model,
                src,
                src_mask,
                max_len,
                start_symbol,
                end_symbol,
                device=device,
            )

            # Save predicted and actual tokens.
            predictions.append(_ids_to_tokens(pred_ids.squeeze(0).tolist(), tgt_vocab))
            references.append(_ids_to_tokens(tgt, tgt_vocab))

    return _corpus_bleu(predictions, references)


@torch.no_grad()
def log_encoder_attention_maps(
    model,
    data_loader,
    src_vocab,
    device,
    wandb_run,
    max_heads=None,
):
    """Log attention heatmaps for one example."""

    if wandb_run is None:
        return

    model.eval()

    # Take the first batch and first sentence.
    batch = next(iter(data_loader))
    src_batch, tgt_batch = _unpack_batch(batch)

    src = src_batch[:1].to(device)
    tgt = tgt_batch[:1].to(device)

    # Use teacher forcing target input just to run model once.
    tgt_input = tgt[:, :-1]

    src_mask = make_src_mask(src, getattr(model, "src_pad_idx", PAD_IDX))
    tgt_mask = make_tgt_mask(tgt_input, getattr(model, "tgt_pad_idx", PAD_IDX))

    # Forward pass stores attention weights inside the attention module.
    _ = model(src, tgt_input, src_mask, tgt_mask)

    # Take attention from the last encoder layer.
    attn = model.encoder.layers[-1].self_attn.attn_weights

    if attn is None:
        return

    # Remove batch dimension.
    # Shape becomes [num_heads, src_len, src_len].
    attn = attn[0].detach().cpu()

    src_ids = src.squeeze(0).detach().cpu().tolist()
    tokens = [_token_from_vocab(src_vocab, i) for i in src_ids]

    if max_heads is not None:
        attn = attn[:max_heads]

    log_payload = {
        "attention/sample_source_tokens": " ".join(tokens)
    }

    for h in range(attn.size(0)):
        fig, ax = plt.subplots(figsize=(7, 6))

        # Plot attention matrix as heatmap.
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

        # Close figure so memory does not keep increasing.
        plt.close(fig)

    wandb_run.log(log_payload)


def _serialise_vocab(vocab):
    # Convert vocab into dictionary so it can be saved in checkpoint.
    if vocab is None:
        return None

    if hasattr(vocab, "to_serializable"):
        return vocab.to_serializable()

    if hasattr(vocab, "stoi") and hasattr(vocab, "itos"):
        return {
            "stoi": dict(vocab.stoi),
            "itos": list(vocab.itos),
        }

    return None


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    path="checkpoint.pt",
):
    """Save everything needed to continue training or run inference."""

    # Make folder if path has a folder.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Save model settings also, so same architecture can be rebuilt later.
    if hasattr(model, "get_config"):
        model_config = model.get_config()
    else:
        model_config = {
            "src_vocab_size": model.src_vocab_size,
            "tgt_vocab_size": model.tgt_vocab_size,
            "d_model": model.d_model,
            "N": model.N,
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "dropout": model.dropout_p,
        }

    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": model_config,
        "src_vocab": _serialise_vocab(getattr(model, "src_vocab", None)),
        "tgt_vocab": _serialise_vocab(getattr(model, "tgt_vocab", None)),
    }

    torch.save(payload, path)


def _safe_torch_load(path, map_location="cpu"):
    # Different PyTorch versions support torch.load slightly differently.
    # This keeps loading safe on local machine and Gradescope.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)

    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
):
    """Load saved model weights and optional training state."""

    checkpoint = _safe_torch_load(
        path,
        map_location=next(model.parameters()).device,
    )

    # Some checkpoints are full dicts, some may directly be state_dicts.
    state = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(state, strict=True)

    # Load vocab info if it is available.
    if hasattr(model, "_load_vocab_metadata") and isinstance(checkpoint, dict):
        model._load_vocab_metadata(checkpoint)

    # Load optimizer only if user passed optimizer.
    if optimizer is not None and isinstance(checkpoint, dict) and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler only if user passed scheduler.
    if scheduler is not None and isinstance(checkpoint, dict) and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if isinstance(checkpoint, dict):
        return int(checkpoint.get("epoch", 0))

    return 0


def run_training_experiment():
    # All command line options are defined here.
    parser = argparse.ArgumentParser(description="Train Transformer on Multi30k German to English")

    # Model/training size settings.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=4000)

    # Dataset/vocab settings.
    parser.add_argument("--max_vocab_size", type=int, default=12000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=100)

    # Checkpoint paths.
    parser.add_argument("--checkpoint", type=str, default="best_checkpoint.pth")
    parser.add_argument("--last_checkpoint", type=str, default="last_checkpoint.pth")

    # Device is GPU if available, else CPU.
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Optimizer and loss settings.
    parser.add_argument("--fixed_lr", action="store_true", help="Use constant lr instead of Noam scheduler")
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=0.1)

    parser.add_argument("--num_workers", type=int, default=2)

    # W&B logging settings.
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="da6401-a3")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--experiment_tag", type=str, default="report")

    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--log_grad_steps", type=int, default=0)
    parser.add_argument("--log_prediction_confidence", action="store_true")
    parser.add_argument("--log_attention_sample", action="store_true")
    parser.add_argument("--compute_val_bleu", action="store_true")

    # Ablation experiment flags.
    parser.add_argument("--no_scale_attention", action="store_true")
    parser.add_argument("--learned_positional", action="store_true")

    args = parser.parse_args()

    # For fixed LR experiment, use 1e-4 if user did not specify LR.
    if args.fixed_lr and args.lr == 1.0:
        args.lr = 1e-4

    # Normal Transformer uses scaling in attention.
    # Only remove scaling for the ablation experiment.
    model_module.USE_ATTENTION_SCALE = not args.no_scale_attention

    device = torch.device(args.device)

    wandb_run = None

    if args.use_wandb:
        import wandb as _wandb

        tags = [args.experiment_tag]

        # Tags help separate runs in W&B report.
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

    # Build train dataset first because vocab should be built only from training data.
    train_ds = Multi30kDataset(
        split="train",
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        max_len=args.max_len,
    )

    src_vocab, tgt_vocab = train_ds.build_vocab()
    train_ds.process_data()

    # Validation and test reuse training vocab.
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

    # DataLoaders create batches.
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

    # BLEU decoding is sentence by sentence, so batch size 1 is simple.
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=test_ds.collate_fn,
        num_workers=0,
    )

    # Create Transformer model.
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

    # Attach vocab to model so inference/checkpoint can use it.
    model.set_vocabs(src_vocab, tgt_vocab)

    # Adam settings are from the Transformer paper.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # Fixed LR run has no scheduler.
    # Normal run uses Noam scheduler.
    if args.fixed_lr:
        scheduler = None
    else:
        scheduler = NoamScheduler(
            optimizer,
            d_model=args.d_model,
            warmup_steps=args.warmup_steps,
        )

    loss_fn = LabelSmoothingLoss(
        len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=args.smoothing,
    )

    best_val = float("inf")
    step_state = {"step": 0}

    for epoch in range(1, args.epochs + 1):
        # Train one epoch.
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            True,
            str(device),
            wandb_run=wandb_run,
            step_state=step_state,
            log_every_steps=args.log_every_steps,
            log_grad_steps=args.log_grad_steps,
            log_prediction_confidence=args.log_prediction_confidence,
        )

        # Validate one epoch.
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch,
            False,
            str(device),
        )

        # Extra validation metric for easier W&B plots.
        val_acc = evaluate_token_accuracy(
            val_loader,
            model,
            str(device),
        )

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_token_acc={val_acc:.4f}"
        )

        # Always save latest checkpoint.
        save_checkpoint(model, optimizer, scheduler, epoch, args.last_checkpoint)

        # Save best checkpoint based on validation loss.
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, args.checkpoint)
            print(f"Saved best checkpoint to {args.checkpoint}")

        # Log epoch metrics to W&B.
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/val_token_accuracy": val_acc,
                    "epoch/best_val_loss": best_val,
                },
                step=step_state["step"],
            )

    # Log attention heatmaps if requested.
    if args.log_attention_sample and wandb_run is not None:
        print("Logging last encoder-layer attention maps to W&B...")
        log_encoder_attention_maps(
            model,
            val_loader,
            src_vocab,
            str(device),
            wandb_run,
        )

    # Use best checkpoint for final BLEU.
    print("Loading best checkpoint for BLEU evaluation...")

    load_checkpoint(
        args.checkpoint,
        model,
        optimizer=None,
        scheduler=None,
    )

    # Validation BLEU is optional because it takes time.
    if args.compute_val_bleu:
        val_bleu = evaluate_bleu(
            model,
            val_loader,
            tgt_vocab,
            device=str(device),
            max_len=args.max_len,
        )

        print(f"Validation BLEU: {val_bleu:.2f}")

        if wandb_run is not None:
            wandb_run.log(
                {"val_bleu": val_bleu},
                step=step_state["step"],
            )

    # Test BLEU is the final translation quality metric.
    bleu = evaluate_bleu(
        model,
        test_loader,
        tgt_vocab,
        device=str(device),
        max_len=args.max_len,
    )

    print(f"Test BLEU: {bleu:.2f}")

    if wandb_run is not None:
        wandb_run.log(
            {"test_bleu": bleu},
            step=step_state["step"],
        )
        wandb_run.finish()


if __name__ == "__main__":
    run_training_experiment()