#!/usr/bin/env python3
"""
sft_train.py — Production-grade SFT training + merge-to-standard-checkpoint
========================================================================
Trains ONE PEFT method at ONE seed via supervised instruction fine-tuning
(Alpaca, prompt-masked loss), then bakes the adapter into the base weights
and saves a STANDARD HuggingFace checkpoint that lm-eval-harness can load
directly. This is what makes the comparison defensible:

    - Same base model, same SFT data, same hyperparameters across methods
      (only the adapter mechanism differs) -> fair.
    - Loss only on the response tokens (instruction masked) -> real SFT,
      not wikitext LM.
    - Output is a plain Llama checkpoint (adapter merged) -> evaluated by a
      recognized third-party harness, not homemade metrics.

Usage:
    python scripts/sft_train.py --method blora --seed 0 \
        --out-dir ~/ckpt/blora_s0 --max-steps 1000 --hf-token $HF_TOKEN
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from blora import BloRAAdapter, freeze_for_blora

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
TARGET_MODS = ["q_proj", "k_proj", "v_proj", "o_proj"]
SEQ_LEN = 512
LR = 2e-4  # SAME for every method (fairness)
WARMUP_FRAC = 0.03
WD = 0.01
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))  # effective batch
MICRO_BS = 1


# ---------------------------------------------------------------- seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
PROMPT_W_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}"
    "\n\n### Response:\n"
)


def build_sft_examples(tok, seed, n_examples):
    """Alpaca SFT, prompt tokens masked (-100), loss only on response."""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n_examples]

    examples = []
    for i in idx:
        ex = ds[i]
        prompt = (PROMPT_W_INPUT if ex["input"].strip() else PROMPT_NO_INPUT).format(
            instruction=ex["instruction"], input=ex["input"]
        )
        response = ex["output"].strip() + tok.eos_token

        p_ids = tok(prompt, add_special_tokens=True, truncation=True, max_length=SEQ_LEN)[
            "input_ids"
        ]
        r_ids = tok(response, add_special_tokens=False, truncation=True, max_length=SEQ_LEN)[
            "input_ids"
        ]
        ids = (p_ids + r_ids)[:SEQ_LEN]
        labels = ([-100] * len(p_ids) + r_ids)[:SEQ_LEN]
        if len(ids) < 4 or all(lab == -100 for lab in labels):
            continue
        examples.append((ids, labels))
    print(f"  SFT examples: {len(examples)} (alpaca, seed={seed})", flush=True)
    return examples


def collate(batch, pad_id):
    maxlen = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, labs in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(labs + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


# ---------------------------------------------------------------- model
def load_base(hf_token):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", dtype=torch.bfloat16, token=hf_token
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model


def attach_adapter(model, method):
    if method in ("blora", "bcms"):
        model.enable_input_require_grads()
        BloRAAdapter.replace_in_model(
            model, target_modules=TARGET_MODS, lora_r=16, n_bases=4, multi_dir=True, verbose=False
        )
        freeze_for_blora(model)
        if method == "bcms":
            # BCMS-only ablation: zero + freeze the LoRA half so only the
            # circulant BCMS subspace is trained. lora_B is already zero-init,
            # so B@A == 0 throughout -> pure BCMS delta.
            for n, p in model.named_parameters():
                if "lora_A" in n or "lora_B" in n:
                    p.data.zero_()
                    p.requires_grad_(False)
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    common = dict(
        target_modules=TARGET_MODS, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
    )
    if method == "lora16":
        cfg = LoraConfig(r=16, lora_alpha=32, **common)
    elif method == "lora":
        cfg = LoraConfig(r=32, lora_alpha=64, **common)
    elif method == "dora":
        cfg = LoraConfig(r=32, lora_alpha=64, use_dora=True, **common)
    elif method == "pissa":
        cfg = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="pissa", **{**common, "lora_dropout": 0.0}
        )
    else:
        raise ValueError(f"unknown method {method}")
    model.enable_input_require_grads()
    return get_peft_model(model, cfg)


# ----------------------------------------------- bake BloRA -> nn.Linear
@torch.no_grad()
def merge_blora_into_model(model):
    """Replace every BloRAAdapter with a plain nn.Linear holding the merged
    weight, so save_pretrained emits a standard checkpoint."""

    def recurse(module):
        for name, child in list(module.named_children()):
            if isinstance(child, BloRAAdapter):
                child.eval()
                child._rebuild_eval_cache()
                W = child._W_merged.to(torch.bfloat16)
                lin = nn.Linear(child.n_in, child.n_out, bias=child.b_frozen is not None)
                lin.weight = nn.Parameter(W.clone(), requires_grad=False)
                if child.b_frozen is not None:
                    lin.bias = nn.Parameter(
                        child.b_frozen.to(torch.bfloat16).clone(), requires_grad=False
                    )
                setattr(module, name, lin)
            else:
                recurse(child)

    recurse(model)
    return model


def to_standard_model(model, method):
    if method == "blora":
        return merge_blora_into_model(model)
    return model.merge_and_unload()  # peft -> base model with merged weights


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method", required=True, choices=["blora", "bcms", "lora16", "lora", "dora", "pissa"]
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--n-examples", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()

    set_seed(args.seed)
    if args.hf_token:
        from huggingface_hub import login

        login(token=args.hf_token, add_to_git_credential=False)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=args.hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    examples = build_sft_examples(tok, args.seed, args.n_examples)
    set_seed(args.seed)
    model = load_base(args.hf_token)
    model = attach_adapter(model, args.method)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  method={args.method} seed={args.seed} trainable={n_train:,} ({n_train / 1e6:.2f}M)",
        flush=True,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=WD
    )
    sched = get_cosine_schedule_with_warmup(opt, int(args.max_steps * WARMUP_FRAC), args.max_steps)

    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id
    model.train()
    order = list(range(len(examples)))
    random.Random(args.seed).shuffle(order)
    ptr = 0
    t0 = time.perf_counter()
    losses = []

    for step in range(args.max_steps):
        opt.zero_grad()
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            batch = []
            for _ in range(MICRO_BS):
                if ptr >= len(order):
                    random.Random(args.seed + step).shuffle(order)
                    ptr = 0
                batch.append(examples[order[ptr]])
                ptr += 1
            input_ids, labels, attn = collate(batch, pad_id)
            input_ids, labels, attn = input_ids.to(dev), labels.to(dev), attn.to(dev)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        losses.append(step_loss)
        if step % 50 == 0 or step == args.max_steps - 1:
            el = (time.perf_counter() - t0) / 60
            print(
                f"  step {step:4d}/{args.max_steps} | loss={step_loss:.4f} | {el:.1f}min",
                flush=True,
            )

    train_min = (time.perf_counter() - t0) / 60
    peak_vram = torch.cuda.max_memory_allocated(0) / 1e9
    print(
        f"  trained {args.max_steps} steps in {train_min:.1f}min | peak VRAM {peak_vram:.2f}GB",
        flush=True,
    )

    # bake adapter -> standard checkpoint
    print("  merging adapter into standard checkpoint...", flush=True)
    model = to_standard_model(model, args.method)
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tok.save_pretrained(args.out_dir)

    meta = {
        "method": args.method,
        "seed": args.seed,
        "base": MODEL_NAME,
        "trainable_params": n_train,
        "trainable_M": round(n_train / 1e6, 2),
        "max_steps": args.max_steps,
        "n_examples": len(examples),
        "lr": args.lr,
        "grad_accum": GRAD_ACCUM,
        "seq_len": SEQ_LEN,
        "effective_batch": GRAD_ACCUM * MICRO_BS,
        "train_min": round(train_min, 1),
        "peak_vram_gb": round(peak_vram, 2),
        "final_loss": round(sum(losses[-50:]) / min(50, len(losses)), 4),
        "date": datetime.now().isoformat(),
    }
    with open(os.path.join(args.out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved -> {args.out_dir}", flush=True)
    print("META " + json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
