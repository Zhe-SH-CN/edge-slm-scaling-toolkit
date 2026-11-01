"""
Hardware Roofline & Decoding Throughput Benchmarking (Proposition 1).
Measures Time-To-First-Token (TTFT), prefill throughput, and batch-1 decoding speed.
"""

import time
from typing import Dict, Any, List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

@torch.no_grad()
def benchmark_batch1_decoding(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 64,
    warmup: int = 2
) -> Dict[str, Any]:
    """
    Evaluates empirical batch-1 latency and throughput under memory-bound Roofline regime.
    """
    device = next(model.parameters()).device

    # Warmup runs
    for p in prompts[:warmup]:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        _ = model.generate(**inputs, max_new_tokens=16, do_sample=False)

    ttft_list = []
    decode_speeds = []
    total_tokens_gen = 0
    total_gen_time = 0.0

    torch.cuda.reset_peak_memory_stats(device)
    static_vram_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        # Step 1: Forward single token to measure TTFT
        outputs = model(inputs["input_ids"])
        torch.cuda.synchronize()
        ttft = (time.perf_counter() - t0) * 1000.0
        ttft_list.append(ttft)

        # Step 2: Autoregressive decoding
        t_gen_start = time.perf_counter()
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False
        )
        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t_gen_start

        gen_tokens = gen_out.shape[1] - input_len
        if gen_tokens > 0:
            speed = gen_tokens / t_gen
            decode_speeds.append(speed)
            total_tokens_gen += gen_tokens
            total_gen_time += t_gen

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "static_vram_mb": round(static_vram_mb, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "mean_ttft_ms": round(sum(ttft_list) / len(ttft_list), 2),
        "decode_tokens_per_sec": round(sum(decode_speeds) / len(decode_speeds), 2),
        "total_generated_tokens": total_tokens_gen
    }
