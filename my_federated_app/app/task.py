"""Task module for model and data preparation."""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name: str = "EleutherAI/gpt-neo-1.3B"):
    """Load LLM with sharding and quantization."""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
        low_cpu_mem_usage=True
    )

def load_tokenizer(model_name: str = "EleutherAI/gpt-neo-1.3B"):
    """Load tokenizer for LLM."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_data():
    """Simulate client-specific training data."""
    return [
        "Federated learning enables collaborative model training",
        "Large language models require distributed training strategies"
    ]