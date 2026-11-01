# Edge SLM Scaling Toolkit ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Technical Report](https://img.shields.io/badge/Technical%20Report-PDF-b31b1b.svg)](paper/main.pdf)

> **English** | [简体中文](README_zh.md)

Official research toolkit for the paper:  
**"Scaling Laws at the Edge: First-Principles Adaptation, Subspace Allocation, and Quantization Trade-offs for Small Language Models"**  
*Sizhe Zhang, Wenyuan Wang, Yuxuan Zhang, Zheyang Dong, Qinqin Zhang\** (Shanghai Jiao Tong University)

---

## 📌 Overview

This toolkit provides an end-to-end framework for parameter-efficient adaptation, subspace specialization, and low-precision deployment of Small Language Models (SLMs, 26M to 1.5B parameters) on memory- and compute-constrained edge accelerators (e.g., NVIDIA Turing sm_75 / Tesla T4).

```text
+-------------------------------------------------------------------------------------------------+
|                                 Edge SLM Scaling Architecture                                    |
+-------------------------------------------------------------------------------------------------+
| 1. Hardware Roofline & Decoding Formulation                                                     |
|    - Batch-1 token streaming: I_decode = 2/b << I* (Memory-bound)                               |
|    - Dequantization latency bound: t_dequant < (2-b) * Phi / B_mem                              |
|                                                                                                 |
| 2. Subspace Allocation Hierarchy                                                                |
|    - Syntactic Schemas (JSONOut, Intent) --> M_qv = {W_q, W_v} (100% saturation)               |
|    - Factual Memory (CourseQA)           --> M_ffn = {W_gate, W_up, W_down} (+13.36% F1 gain)     |
|                                                                                                 |
| 3. Spectral Rank Truncation (Intrinsic Dimension)                                               |
|    - Rank-4 LoRA acts as an implicit regularizer, outperforming unconstrained Full SFT by +8.9% |
|                                                                                                 |
| 4. DPO Likelihood Displacement Phase Transition                                                  |
|    - Step size bound: eta <= 5e-6 stabilizes log-ratio trust region, preventing token collapse   |
+-------------------------------------------------------------------------------------------------+
```

---

## 🏆 Key Findings & Empirical Benchmarks

Based on 39 controlled runs across 4 model scales (`MiniMind2-26M`, `MiniMind2-104M`, `Qwen2.5-0.5B`, `Qwen2.5-1.5B`) and 3 zero-leakage tasks:

1. **LoRA Beats Full SFT on Small Corpora**: On `MiniMind-26M`, rank-16 LoRA reaches **45.71%** overall on CourseQA vs. **36.80%** for 8-epoch Full SFT (+8.91% absolute gain) while cutting peak VRAM by **65.1%**.
2. **Subspace Specialization Dichotomy**: Adapting `W_q, W_v` saturates strict JSON schema extraction (**100.0%**), but factual domain QA requires adapting SwiGLU MLP layers (**32.23%** vs. 18.87% for `qv`).
3. **Batch-1 Decoding Roofline (T4 sm_75)**: 
   - `LLM.int8()` is strictly dominated: outlier splitting overhead drops throughput to **5.77 tok/s** (3.9x slower than FP16).
   - `4-bit NF4` double quantization achieves **16.01 tok/s** (2.77x faster than INT8) while saving **505.6 MB** static VRAM.
4. **DPO Step-size Transition**: Bounding DPO learning rate from $5\times 10^{-5}$ to $5\times 10^{-6}$ eliminates off-manifold likelihood displacement, doubling classification accuracy on Intent.

---

## 📂 Repository Structure

```text
.
├── data/                    # 1,770 verified zero-leakage benchmark samples
│   ├── courseqa/            # Multi-hop technical QA (8 domains)
│   ├── jsonout/             # Unstructured text to strict schema JSON
│   └── intent/              # Out-of-template 6-way intent classification
├── results/                 # Raw JSON evaluation metrics from all 39 ablation runs
├── slm_toolkit/             # Modular, clean Python implementation
│   ├── models.py            # Hardware-aware precision loader (FP16, INT8, NF4)
│   ├── data_loader.py       # Assistant-only loss masking collator
│   ├── train.py             # SFT, LoRA, and QLoRA engine with subspace routing
│   ├── roofline.py          # Batch-1 TTFT and throughput benchmarking
│   └── evaluate.py          # Dispatched evaluation suite (EM, Token-F1, JSON Validity)
├── scripts/
│   ├── plot_figures.py      # Reproduce Figure 1, 2, 3 from raw results
│   └── run_benchmarks.sh    # End-to-end evaluation pipeline
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Zhe-SH-CN/edge-slm-scaling-toolkit.git
cd edge-slm-scaling-toolkit
pip install -r requirements.txt
```

### 2. Run Batch-1 Roofline Benchmark (Proposition 1)

```python
from slm_toolkit.models import load_model_and_tokenizer
from slm_toolkit.roofline import benchmark_batch1_decoding

# Load Qwen2.5-0.5B in 4-bit NF4
model, tokenizer = load_model_and_tokenizer(
    "Qwen/Qwen2.5-0.5B-Instruct", 
    quantization="nf4"
)

prompts = ["What is the time complexity of QuickSort in the average case?"]
stats = benchmark_batch1_decoding(model, tokenizer, prompts, max_new_tokens=64)
print(stats)
# Output: {'static_vram_mb': 2207.1, 'decode_tokens_per_sec': 16.01, ...}
```

### 3. Fine-tuning with Subspace Allocation

```python
from slm_toolkit.models import load_model_and_tokenizer
from slm_toolkit.train import train_slm

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

# Train with full attention + SwiGLU MLP adaptation for factual QA
train_slm(
    model=model,
    tokenizer=tokenizer,
    train_jsonl="data/courseqa/train.jsonl",
    dev_jsonl="data/courseqa/dev.jsonl",
    method="lora",
    subspace="all-linear",
    lora_r=4,
    epochs=6
)
```

### 4. Reproduce Paper Figures

```bash
python scripts/plot_figures.py
```

---

## 📜 Citation

```bibtex
@article{zhang2026slmedge,
  title={Scaling Laws at the Edge: First-Principles Adaptation, Subspace Allocation, and Quantization Trade-offs for Small Language Models},
  author={Zhang, Sizhe and Wang, Wenyuan and Zhang, Yuxuan and Dong, Zheyang and Zhang, Qinqin},
  journal={arXiv preprint arXiv:2609.xxxxx},
  year={2026}
}
```

## 📄 License

This repository is distributed under the [MIT License](LICENSE).
