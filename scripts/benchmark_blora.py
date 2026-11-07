#!/usr/bin/env python3
"""
BloRA vs 2026 PEFT Methods — Llama-3-8B Benchmark
==================================================
Model:    meta-llama/Meta-Llama-3-8B (bf16)
Hardware: 1x NVIDIA L4 (24GB)
Methods:  BloRA, LoRA, DoRA, PiSSA, rsLoRA, VeRA
Dataset:  wikitext-2 (train=full, test=held-out PPL)
Eval:     Test PPL + HellaSwag accuracy
Steps:    1000

Usage:
    python scripts/benchmark_blora.py --hf-token YOUR_TOKEN
    python scripts/benchmark_blora.py --hf-token YOUR_TOKEN --steps 500
    python scripts/benchmark_blora.py --hf-token YOUR_TOKEN --methods blora lora
"""

import argparse
import gc
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from blora import BloRAAdapter, freeze_for_blora

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
STEPS = 1000
BATCH_SIZE = 1
SEQ_LEN = 512
LR = 2e-4
LORA_R = 32
LORA_ALPHA = 64
TARGET_MODS = ["q_proj", "k_proj", "v_proj", "o_proj"]
OUT_FILE = "benchmark_results.json"


def gpu_info():
    n = torch.cuda.device_count()
    total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n))
    names = [torch.cuda.get_device_properties(i).name for i in range(n)]
    return n, total / 1e9, names


def reset_peak():
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def peak_vram_gb():
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def free_model(model):
    torch.cuda.synchronize()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_base(hf_token=None):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    model.gradient_checkpointing_enable()
    return model, tok


def get_train_batches(tok):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x for x in ds["text"] if len(x.strip()) > 200]
    print(f"  Train samples: {len(texts)} (full wikitext-2 train)")
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=SEQ_LEN)
    return enc["input_ids"]


@torch.no_grad()
def eval_test_ppl(model, tok):
    print("  Evaluating test PPL...", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x for x in ds["text"] if len(x.strip()) > 200][:500]
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=SEQ_LEN)
    input_ids = enc["input_ids"]

    model.eval()
    dev = next(model.parameters()).device
    total_loss, n_batches = 0.0, 0

    for i in range(0, len(input_ids), 4):
        batch = input_ids[i : i + 4].to(dev)
        out = model(input_ids=batch, labels=batch)
        total_loss += out.loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    ppl = math.exp(min(avg_loss, 20))
    print(f"  Test PPL: {ppl:.2f}")
    return round(ppl, 2)


@torch.no_grad()
def eval_hellaswag(model, tok, n_samples=500):
    print(f"  Evaluating HellaSwag ({n_samples} samples)...", flush=True)
    ds = load_dataset("Rowan/hellaswag", split="validation").select(range(n_samples))
    model.eval()
    dev = next(model.parameters()).device
    correct = 0

    for ex in ds:
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])

        losses = []
        for ending in endings:
            text = ctx + " " + ending
            enc = tok(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN).to(dev)
            ids = enc["input_ids"]
            with torch.no_grad():
                out = model(input_ids=ids, labels=ids)
            losses.append(out.loss.item())

        pred = losses.index(min(losses))
        if pred == label:
            correct += 1

    acc = correct / n_samples * 100
    print(f"  HellaSwag accuracy: {acc:.1f}%")
    return round(acc, 1)


def run_benchmark(method_name, model, tok, input_ids, lr=None, warmup_steps=None):
    set_seed()
    print(f"\n{'=' * 65}")
    print(f"  {method_name}")
    print(f"{'=' * 65}")

    n_train = trainable_params(model)
    print(f"  Trainable params: {n_train:,} ({n_train / 1e6:.2f}M)")
    if n_train == 0:
        raise RuntimeError("No trainable parameters — base model not properly set up")

    _lr = lr if lr is not None else LR
    _warmup = warmup_steps if warmup_steps is not None else max(1, STEPS // 10)
    print(f"  LR: {_lr:.0e} | Warmup: {_warmup} steps")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=_lr,
        weight_decay=0.01,
    )
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=_warmup,
        num_training_steps=STEPS,
    )

    reset_peak()
    model.train()
    losses, step_times = [], []
    n_batches = input_ids.shape[0]
    t_train_start = time.perf_counter()

    try:
        for step in range(STEPS):
            batch = input_ids[step % n_batches : step % n_batches + BATCH_SIZE]
            batch = batch.to(next(model.parameters()).device)

            t0 = time.perf_counter()
            opt.zero_grad()
            out = model(input_ids=batch, labels=batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step_times.append((time.perf_counter() - t0) * 1000)
            losses.append(out.loss.item())

            if step % 100 == 0 or step == STEPS - 1:
                ppl = math.exp(min(losses[-1], 20))
                elapsed = (time.perf_counter() - t_train_start) / 60
                print(
                    f"  step {step:4d}/{STEPS} | loss={losses[-1]:.4f} | "
                    f"ppl={ppl:.2f} | {step_times[-1]:.0f}ms | {elapsed:.1f}min"
                )

        train_total_min = (time.perf_counter() - t_train_start) / 60

        model.eval()
        dev = next(model.parameters()).device
        warmup = input_ids[:1].to(dev)
        with torch.no_grad():
            for _ in range(3):
                model(input_ids=warmup)
            t0 = time.perf_counter()
            for _ in range(10):
                model(input_ids=warmup)
            inf_ms = (time.perf_counter() - t0) / 10 * 1000

        final_loss = sum(losses[-50:]) / min(50, len(losses))
        train_ppl = round(math.exp(min(final_loss, 20)), 2)
        peak_vram = round(peak_vram_gb(), 2)
        avg_step = round(sum(step_times[20:]) / max(1, len(step_times) - 20), 1)

        print(
            f"\n  Training done: {train_total_min:.1f} min | "
            f"Train PPL: {train_ppl:.2f} | VRAM: {peak_vram:.2f}GB"
        )

        test_ppl = eval_test_ppl(model, tok)
        hellaswag = eval_hellaswag(model, tok, n_samples=500)

        result = {
            "method": method_name,
            "train_ppl": train_ppl,
            "test_ppl": test_ppl,
            "hellaswag_acc": hellaswag,
            "vram_gb": peak_vram,
            "train_ms_per_step": avg_step,
            "train_total_min": round(train_total_min, 1),
            "inf_ms": round(inf_ms, 1),
            "params_M": round(n_train / 1e6, 2),
            "loss_curve": [round(v, 4) for v in losses[::100]],
            "status": "OK",
        }
        del opt, sched
        gc.collect()
        torch.cuda.empty_cache()
        return result

    except torch.cuda.OutOfMemoryError:
        print("  OOM")
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "method": method_name,
            "train_ppl": None,
            "test_ppl": None,
            "hellaswag_acc": None,
            "vram_gb": None,
            "train_ms_per_step": None,
            "train_total_min": None,
            "inf_ms": None,
            "params_M": round(n_train / 1e6, 2),
            "status": "OOM",
        }


def bench_lora(hf_token, input_ids, tok):
    from peft import LoraConfig, TaskType, get_peft_model

    model, _ = load_base(hf_token)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODS,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    result = run_benchmark("LoRA (r=32)", model, tok, input_ids)
    free_model(model)
    return result


def bench_lora_r16(hf_token, input_ids, tok):
    from peft import LoraConfig, TaskType, get_peft_model

    model, _ = load_base(hf_token)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=TARGET_MODS,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    result = run_benchmark("LoRA (r=16)", model, tok, input_ids)
    free_model(model)
    return result


def bench_dora(hf_token, input_ids, tok):
    from peft import LoraConfig, TaskType, get_peft_model

    model, _ = load_base(hf_token)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODS,
            use_dora=True,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    result = run_benchmark("DoRA (r=32)", model, tok, input_ids)
    free_model(model)
    return result


def bench_pissa(hf_token, input_ids, tok):
    from peft import LoraConfig, TaskType, get_peft_model

    model, _ = load_base(hf_token)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODS,
            init_lora_weights="pissa",
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    result = run_benchmark("PiSSA (r=32)", model, tok, input_ids)
    free_model(model)
    return result


def bench_rslora(hf_token, input_ids, tok):
    from peft import LoraConfig, TaskType, get_peft_model

    model, _ = load_base(hf_token)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODS,
            use_rslora=True,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    result = run_benchmark("rsLoRA (r=32)", model, tok, input_ids)
    free_model(model)
    return result


def bench_vera(hf_token, input_ids, tok):
    from peft import TaskType, VeraConfig, get_peft_model

    model, _ = load_base(hf_token)
    model = get_peft_model(
        model,
        VeraConfig(
            r=256,
            target_modules=TARGET_MODS,
            vera_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    result = run_benchmark("VeRA (r=256)", model, tok, input_ids)
    free_model(model)
    return result


def _bench_blora(hf_token, input_ids, tok, lora_r, n_bases, label):
    model, _ = load_base(hf_token)
    model.enable_input_require_grads()
    reps = BloRAAdapter.replace_in_model(
        model,
        target_modules=TARGET_MODS,
        lora_r=lora_r,
        n_bases=n_bases,
        multi_dir=True,
        verbose=False,
    )
    st = freeze_for_blora(model)
    print(
        f"  {label}: {len(reps)} layers | trainable {st['trainable']:,} "
        f"(LoRA r={lora_r} + BCMS hvd k={n_bases})"
    )
    result = run_benchmark(label, model, tok, input_ids, lr=3e-4, warmup_steps=50)
    free_model(model)
    return result


def bench_blora(hf_token, input_ids, tok):
    return _bench_blora(hf_token, input_ids, tok, lora_r=16, n_bases=4, label="BloRA (r16+k4)")


def bench_blora_r8(hf_token, input_ids, tok):
    return _bench_blora(hf_token, input_ids, tok, lora_r=8, n_bases=4, label="BloRA (r8+k4)")


METHODS = {
    "blora": bench_blora,
    "blora_r8": bench_blora_r8,
    "lora": bench_lora,
    "lora_r16": bench_lora_r16,
    "dora": bench_dora,
    "pissa": bench_pissa,
    "rslora": bench_rslora,
    "vera": bench_vera,
}


def print_table(results, n_gpu, total_vram, gpu_names):
    print(f"\n{'=' * 90}")
    print("  BloRA vs 2026 PEFT State-of-the-Art — Benchmark")
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Hardware: {n_gpu}x {gpu_names[0]} ({total_vram:.0f}GB)")
    print("  Dataset:  wikitext-2 full train | test PPL + HellaSwag eval")
    print(f"  Steps:    {STEPS} | Batch: {BATCH_SIZE} | Samples: full")
    print(f"  Date:     {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'=' * 90}")
    print(
        f"  {'Method':<20} {'Train PPL':>10} {'Test PPL':>9} {'HellaSwag':>10} "
        f"{'VRAM':>7} {'ms/step':>8} {'Total min':>10} {'Params':>9}"
    )
    print(f"  {'-' * 88}")

    for r in results:
        if r["status"] != "OK":
            label = "OOM" if r["status"] == "OOM" else "ERR"
            row = (
                f"  {r['method']:<20} {label:>10} {label:>9} {label:>10} "
                f"{label:>7} {label:>8} {label:>10} {r['params_M']:>8.1f}M"
            )
        else:
            row = (
                f"  {r['method']:<20} {r['train_ppl']:>10.2f} {r['test_ppl']:>9.2f} "
                f"{r['hellaswag_acc']:>9.1f}% {r['vram_gb']:>6.1f}G "
                f"{r['train_ms_per_step']:>8.0f} {r['train_total_min']:>9.1f}m "
                f"{r['params_M']:>8.1f}M"
            )
        print(row)

    print(f"{'=' * 90}\n")


def main():
    set_seed()
    global STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    args = ap.parse_args()
    STEPS = args.steps

    if args.hf_token:
        from huggingface_hub import login

        login(token=args.hf_token, add_to_git_credential=False)

    n_gpu, total_vram, gpu_names = gpu_info()
    print(f"\nHardware: {n_gpu}x {gpu_names[0]} | {total_vram:.1f}GB")
    print(f"Model:    {MODEL_NAME}")
    print(f"Steps:    {STEPS} | Batch: {BATCH_SIZE} | SeqLen: {SEQ_LEN}\n")

    print("Loading tokenizer + dataset...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=args.hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    input_ids = get_train_batches(tok)
    print(f"Ready: {input_ids.shape[0]} train batches\n")

    results = []
    t_total = time.perf_counter()

    for name in args.methods:
        if name not in METHODS:
            print(f"Unknown method: {name}")
            continue
        try:
            r = METHODS[name](args.hf_token, input_ids, tok)
        except Exception as e:
            print(f"ERROR {name}: {e}")
            r = {
                "method": name,
                "status": f"ERROR: {str(e)[:80]}",
                "train_ppl": None,
                "test_ppl": None,
                "hellaswag_acc": None,
                "vram_gb": None,
                "train_ms_per_step": None,
                "train_total_min": None,
                "inf_ms": None,
                "params_M": 0,
            }
        results.append(r)
        with open(OUT_FILE, "w") as f:
            json.dump(
                {
                    "model": MODEL_NAME,
                    "hardware": f"{n_gpu}x{gpu_names[0]}",
                    "date": datetime.now().isoformat(),
                    "steps": STEPS,
                    "samples": "full_wikitext2",
                    "results": results,
                },
                f,
                indent=2,
            )

    total_hours = (time.perf_counter() - t_total) / 3600
    print_table(results, n_gpu, total_vram, gpu_names)
    print(f"Total benchmark time: {total_hours:.1f} hours")
    print(f"Results saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
