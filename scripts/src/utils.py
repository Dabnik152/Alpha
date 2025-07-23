from peft import get_peft_model
from src.constants import DEVICE
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def get_model(model_path, to_device=False):
    if model_path == "meta-llama/Meta-Llama-3-8B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16, use_safetensors=True)
        if to_device:
            model.to(DEVICE)
        
    elif model_path == "meta-llama/Llama-3.1-70B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype="auto")
        
    else:
        token = os.getenv("HF_TOKEN")
        model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir="/home/morg/dataset/models/", token=token)
        if to_device:
            model.to(DEVICE)

    return model


def get_peft_pt_model(model_path, peft_config):
    model = get_model(model_path)
    return get_peft_model(model, peft_config)


def get_tokenizer(model_path):
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/home/morg/dataset/models/", use_auth_token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer