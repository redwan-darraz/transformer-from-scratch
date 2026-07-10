"""
The three tokenizers tok.py compares: our BPE from lab01, tiktoken and HuggingFace.
"""

import os

import tiktoken
from tokenizers import Tokenizer

import bpe

TIKTOKEN_ENCODING = "cl100k_base"
HF_TOKENIZER_FILE = os.path.join(os.path.dirname(__file__), "mistral_tokenizer.json")

# Cached lazily on first use — loading is the slowest step (~0.7s), so we
# pay it once per process instead of once per call (e.g. main output + --cost).
_tiktoken_enc = None
_hf_tokenizer = None


def run_bpe(text):
    return bpe.tokenize(text)


def run_tiktoken(text):
    global _tiktoken_enc
    if _tiktoken_enc is None:
        _tiktoken_enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return [_tiktoken_enc.decode([tid]) for tid in _tiktoken_enc.encode(text)]


def run_huggingface(text):
    global _hf_tokenizer
    if _hf_tokenizer is None:
        _hf_tokenizer = Tokenizer.from_file(HF_TOKENIZER_FILE)
    return _hf_tokenizer.encode(text, add_special_tokens=False).tokens


TOKENIZERS = [
    ("BPE (from scratch)", run_bpe),
    ("tiktoken (cl100k_base)", run_tiktoken),
    ("HuggingFace (mistral)", run_huggingface),
]
