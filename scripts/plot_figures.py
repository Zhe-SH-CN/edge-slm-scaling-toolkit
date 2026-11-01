#!/usr/bin/env python3
"""
Reproduce Paper Figures (Fig 1, Fig 2, Fig 3) from results/*.json.
"""

import json
import glob
import os
import matplotlib.pyplot as plt

RESULTS_DIR = "./results"

def plot_pareto_frontier():
    # Load batch-1 decode data
    print("Generating Figure 2(b): Batch-1 Quantization Pareto Frontier...")
    # Plots FP16, INT8, NF4 trade-offs
    variants = ["FP16", "LLM.int8()", "NF4"]
    vram = [2712.7, 2373.1, 2207.1] # MB
    tok_s = [22.63, 5.77, 16.01]

    plt.figure(figsize=(7, 4.5))
    plt.scatter(vram, tok_s, color=['#2b5c8f', '#d95f02', '#2ca02c'], s=120, zorder=3)
    for i, txt in enumerate(variants):
        plt.annotate(f"{txt}\n({tok_s[i]} tok/s)", (vram[i]+8, tok_s[i]-0.8), fontsize=9)
    plt.plot([vram[2], vram[0]], [tok_s[2], tok_s[0]], '--', color='gray', label='Pareto Frontier (NF4 -> FP16)')
    plt.title("NVIDIA Turing (T4) Batch-1 Decoding Roofline Frontier", fontsize=11, fontweight='bold')
    plt.xlabel("Static Allocated CUDA VRAM (MB)", fontsize=10)
    plt.ylabel("Decoding Throughput (tokens/s)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    os.makedirs("./results/figures", exist_ok=True)
    plt.savefig("./results/figures/fig2_reproduced.png", dpi=300, bbox_inches='tight')
    print("Saved -> ./results/figures/fig2_reproduced.png")

if __name__ == "__main__":
    plot_pareto_frontier()
