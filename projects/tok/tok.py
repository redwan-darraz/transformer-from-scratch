"""
tok — compare BPE from-scratch, tiktoken and a HuggingFace tokenizer on any text.
Usage: python tok.py "your text here"
"""

import argparse
import sys

import tiktoken
from transformers import AutoTokenizer

import bpe

sys.stdout.reconfigure(encoding="utf-8")

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


def main():
    parser = argparse.ArgumentParser(
        prog="tok",
        description="Tokenize text and compare BPE (from scratch), tiktoken and a HuggingFace tokenizer.",
    )
    parser.add_argument("text", help="Text to tokenize")
    args = parser.parse_args()

    print(f'Input: "{args.text}"\n')

    for name, fn in [
        ("BPE (from scratch)", run_bpe),
        ("tiktoken (cl100k_base)", run_tiktoken),
        ("HuggingFace (mistral)", run_huggingface),
    ]:
        tokens = fn(args.text)
        print(f"{name} — {len(tokens)} tokens")
        print(f"  {tokens}\n")


if __name__ == "__main__":
    main()
