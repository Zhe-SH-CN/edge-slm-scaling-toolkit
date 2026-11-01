"""
Unified Model Loading & Quantization Setup.
Handles MiniMind2 (26M/104M) and Qwen2.5 (0.5B/1.5B) with proper dtypes and device placement.
"""

from typing import Tuple, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

MODEL_REGISTRY = {
    "minimind2-26m": "jingyaogong/minimind2-small",
    "minimind2-104m": "jingyaogong/minimind2",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct"
}

def load_model_and_tokenizer(
    model_name_or_path: str,
    quantization: Optional[str] = None, # None, 'int8', 'nf4'
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads causal LM and tokenizer with explicit hardware-aware precision settings.
    """
    bnb_config = None
    if quantization == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    elif quantization == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    return model, tokenizer
