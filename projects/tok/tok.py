"""
tok — compare BPE from-scratch, tiktoken and a HuggingFace tokenizer on any text.
Usage: python tok.py "your text here"
"""

import argparse
import logging
import sys

import tiktoken
from colorama import Back, Style, init as colorama_init
from transformers import AutoTokenizer

import bpe

sys.stdout.reconfigure(encoding="utf-8")
colorama_init(autoreset=True)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

TIKTOKEN_ENCODING = "cl100k_base"
HF_MODEL = "mistralai/Mistral-7B-v0.1"

# Cycle of background colors used to tell consecutive tokens apart
TOKEN_COLORS = [Back.RED, Back.GREEN, Back.YELLOW, Back.BLUE, Back.MAGENTA, Back.CYAN]

# Input pricing, USD per 1M tokens (2026 published rates).
# gpt4o / gpt4o-mini are counted with tiktoken (their real tokenizer).
# mistral is counted with the HuggingFace mistral tokenizer (its real tokenizer).
# claude-haiku has no public pip tokenizer, so tiktoken is used as an approximation.
PRICING = {
    "gpt4o":        {"tokenizer": "tiktoken",     "usd_per_million": 2.50},
    "gpt4o-mini":   {"tokenizer": "tiktoken",     "usd_per_million": 0.15},
    "mistral":      {"tokenizer": "huggingface",  "usd_per_million": 2.00},
    "claude-haiku": {"tokenizer": "tiktoken",     "usd_per_million": 0.80},
}


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


def estimate_cost(text, model):
    """Estimated cost in USD cents for tokenizing `text` as input to `model`."""
    info = PRICING[model]
    tokens = run_tiktoken(text) if info["tokenizer"] == "tiktoken" else run_huggingface(text)
    usd = len(tokens) / 1_000_000 * info["usd_per_million"]
    return len(tokens), usd * 100


def main():
    parser = argparse.ArgumentParser(
        prog="tok",
        description="Tokenize text and compare BPE (from scratch), tiktoken and a HuggingFace tokenizer.",
    )
    parser.add_argument("text", help="Text to tokenize")
    parser.add_argument("--cost", action="store_true", help="Show estimated API cost for --model")
    parser.add_argument("--model", choices=list(PRICING.keys()), default="gpt4o",
                         help="Model used to estimate cost with --cost (default: gpt4o)")
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

    if args.cost:
        n_tokens, cents = estimate_cost(args.text, args.model)
        print(f"Estimated cost on {args.model} ({n_tokens} tokens): {cents:.4f} cents (${cents / 100:.6f})")


if __name__ == "__main__":
    main()
