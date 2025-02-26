import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model
