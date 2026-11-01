# Edge SLM Scaling Toolkit (边缘轻量大模型评测与适配套件) ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.xxxxx-b31b1b.svg)](https://arxiv.org/)

> [English](README.md) | **简体中文**

论文官方开源评测与微调套件：  
**《Scaling Laws at the Edge: First-Principles Adaptation, Subspace Allocation, and Quantization Trade-offs for Small Language Models》**  
张思哲、王文远、张宇轩、董哲扬、张勤琴\*（上海交通大学机械与动力工程学院）

---

## 📌 项目定位

本套件针对算力与显存受限的边缘单卡环境（如 NVIDIA Tesla T4 / Turing 架构），系统化提供小模型（26M 至 1.5B）的第一性原理微调适配、子空间分配、低比特量化 Roofline 测速以及偏好对齐稳定性解决方案。

---

## 🏆 核心实证结论 (39 组受控实验)

1. **小样本下 LoRA 反超全参数微调（谱正则化效应）**：在 `MiniMind-26M` 上，r=16 的 LoRA 评测得分达到 **45.71%**，超越 8 轮全参数微调的 **36.80%**（绝对提升 +8.91%），同时训练峰值显存降低 **65.1%**。
2. **子空间分配两极分化**：
   - 模式与语法抽取任务（JSONOut）仅微调注意力 `W_q, W_v` 即可达到 **100.0%** 准确率；
   - 知识记忆类问答（CourseQA）必须放开 SwiGLU MLP 前馈层，得分从 18.87% 跃升至 **32.23%**（提升 +13.36%）。
3. **单批次（Batch-1）解码 Roofline 铁律**：
   - 在 Turing 架构上，`LLM.int8()` 因动态离群通道切分与同步开销，吞吐暴跌至 **5.77 tok/s**（比原生 FP16 慢近 4 倍）；
   - `4-bit NF4` 双重精细量化达到 **16.01 tok/s**（比 INT8 快 2.77 倍），且省下 **505.6 MB** 显存，与 FP16 共同构成边缘部署的最优 Pareto 前沿。
4. **小模型 DPO 学习率相变**：将学习率从 $5\times 10^{-5}$ 约束至 $5\times 10^{-6}$，彻底消除负梯度引起的流形外概率塌缩，分类准确率翻倍。

---

## 📂 模块结构

- `data/`：包含 CourseQA、JSONOut、Intent 三大无泄露基准集（共 1,770 样本）；
- `results/`：完整归档 39 次消融实验原始 JSON 评测记录；
- `slm_toolkit/`：模块化工程包（模型加载、掩码整理器、训练引擎、Roofline 探针、评测协议）；
- `scripts/`：论文图表生成与评测复现脚本。

---

## 📄 开源许可

本项目遵循 [MIT License](LICENSE)。
