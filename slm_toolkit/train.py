"""
Unified LoRA, QLoRA, and Full SFT Trainer for Small Language Models.
"""

from typing import Dict, Any, Optional
import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .data_loader import InstructionDataset, AssistantMaskedDataCollator

# Subspace target definition
TARGET_MODULES = {
    "qv": ["q_proj", "v_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all-linear": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}

def train_slm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_jsonl: str,
    dev_jsonl: Optional[str] = None,
    output_dir: str = "./results/ckpt",
    method: str = "lora", # 'lora', 'qlora', 'full'
    subspace: str = "all-linear", # 'qv', 'qkvo', 'all-linear'
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 6,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    gradient_accumulation_steps: int = 2,
    max_length: int = 512
) -> PreTrainedModel:
    """
    Executes parameter-efficient adaptation or full SFT with assistant-masked loss.
    """
    train_dataset = InstructionDataset(train_jsonl)
    eval_dataset = InstructionDataset(dev_jsonl) if dev_jsonl else None
    data_collator = AssistantMaskedDataCollator(tokenizer, max_length=max_length)

    if method in ("lora", "qlora"):
        if method == "qlora":
            model = prepare_model_for_kbit_training(model)
        
        target_mods = TARGET_MODULES.get(subspace, TARGET_MODULES["all-linear"])
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_mods,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="no",
        evaluation_strategy="no" if not eval_dataset else "epoch",
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    trainer.train()
    return model
