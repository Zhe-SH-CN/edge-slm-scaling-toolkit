"""
Dataset Preparation & Assistant-Only Loss Masking Collator.
"""

from typing import Dict, Any, List
import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class InstructionDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class AssistantMaskedDataCollator:
    """
    Collate function that sets labels to -100 for all system/user prompt tokens,
    computing loss solely on the assistant target completion.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        formatted_texts = []
        target_texts = []

        for item in batch:
            messages = item.get("messages", [])
            # Format using chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                full_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                # Fallback manual formatting
                full_text = ""
                for m in messages:
                    full_text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            
            formatted_texts.append(full_text)
            # Find target response
            target_str = ""
            for m in reversed(messages):
                if m["role"] == "assistant":
                    target_str = m["content"]
                    break
            target_texts.append(target_str)

        encodings = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = encodings["input_ids"].clone()

        # Mask user and system tokens
        for i, target_str in enumerate(target_texts):
            target_ids = self.tokenizer.encode(target_str, add_special_tokens=False)
            input_ids = encodings["input_ids"][i].tolist()
            
            # Find sub-sequence match for target_ids
            matched = False
            for start_idx in range(len(input_ids) - len(target_ids) + 1):
                if input_ids[start_idx : start_idx + len(target_ids)] == target_ids:
                    labels[i, :start_idx] = -100
                    matched = True
                    break
            
            if not matched:
                # Fallback: search for assistant delimiter token
                delim_tokens = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
                for start_idx in range(len(input_ids) - len(delim_tokens) + 1):
                    if input_ids[start_idx : start_idx + len(delim_tokens)] == delim_tokens:
                        labels[i, : start_idx + len(delim_tokens)] = -100
                        break

        # Always mask padding tokens
        labels[encodings["input_ids"] == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        }
