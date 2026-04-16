import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from nnsight import LanguageModel


def load_model_and_tokenizer(checkpoint_path: str, revision: str = None):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, revision=revision, torch_dtype=torch.bfloat16, max_length=2048
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    print("Tokenizer loaded manually, not from model_type")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), tokenizer


def load_nnsight_model_and_tokenizer(checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = 2048
    model = LanguageModel(
        checkpoint_path, tokenizer=tokenizer, device_map="cuda", dispatch=True
    )
    model.eval()

    return model
