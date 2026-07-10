"""
The three tokenizers tok.py compares: our BPE from lab01, tiktoken and HuggingFace.
"""

import logging

import tiktoken
from transformers import AutoTokenizer

import bpe

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

TIKTOKEN_ENCODING = "cl100k_base"
HF_MODEL = "mistralai/Mistral-7B-v0.1"


def run_bpe(text):
    return bpe.tokenize(text)


def run_tiktoken(text):
    enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return [enc.decode([tid]) for tid in enc.encode(text)]


def run_huggingface(text):
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    return tok.tokenize(text)


TOKENIZERS = [
    ("BPE (from scratch)", run_bpe),
    ("tiktoken (cl100k_base)", run_tiktoken),
    ("HuggingFace (mistral)", run_huggingface),
]
