"""
tok — compare BPE from-scratch, tiktoken and a HuggingFace tokenizer on any text.
Usage: python tok.py "your text here"
"""

import argparse
import sys

import tiktoken
from colorama import Back, Style, init as colorama_init
from transformers import AutoTokenizer

import bpe

sys.stdout.reconfigure(encoding="utf-8")
colorama_init(autoreset=True)

TIKTOKEN_ENCODING = "cl100k_base"
HF_MODEL = "mistralai/Mistral-7B-v0.1"

# Cycle of background colors used to tell consecutive tokens apart
TOKEN_COLORS = [Back.RED, Back.GREEN, Back.YELLOW, Back.BLUE, Back.MAGENTA, Back.CYAN]


def run_bpe(text):
    return bpe.tokenize(text)


def run_tiktoken(text):
    enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return [enc.decode([tid]) for tid in enc.encode(text)]


def run_huggingface(text):
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    return tok.tokenize(text)


def render_colored(tokens):
    """Print tokens with alternating background colors so you can see the split visually."""
    parts = []
    for i, token in enumerate(tokens):
        color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
        parts.append(f"{color}{token}{Style.RESET_ALL}")
    print("  " + "".join(parts))


def compression_ratio(text, tokens):
    """Characters per token — higher means the tokenizer compresses better."""
    if not tokens:
        return 0.0
    return len(text) / len(tokens)


def main():
    parser = argparse.ArgumentParser(
        prog="tok",
        description="Tokenize text and compare BPE (from scratch), tiktoken and a HuggingFace tokenizer.",
    )
    parser.add_argument("text", help="Text to tokenize")
    args = parser.parse_args()

    print(f'Input: "{args.text}"  ({len(args.text)} characters)\n')

    for name, fn in [
        ("BPE (from scratch)", run_bpe),
        ("tiktoken (cl100k_base)", run_tiktoken),
        ("HuggingFace (mistral)", run_huggingface),
    ]:
        tokens = fn(args.text)
        ratio = compression_ratio(args.text, tokens)
        print(f"{name} — {len(tokens)} tokens — {ratio:.2f} chars/token")
        render_colored(tokens)
        print()


if __name__ == "__main__":
    main()
