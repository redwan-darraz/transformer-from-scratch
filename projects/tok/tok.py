"""
tok — compare BPE from-scratch, tiktoken and a HuggingFace tokenizer on any text.
Usage: python tok.py "your text here" [--compare] [--cost --model gpt4o]
"""

import argparse
import sys

from display import compression_ratio, render_colored, render_table
from engines import TOKENIZERS
from pricing import PRICING, estimate_cost

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        prog="tok",
        description="Tokenize text and compare BPE (from scratch), tiktoken and a HuggingFace tokenizer.",
    )
    parser.add_argument("text", help="Text to tokenize")
    parser.add_argument("--cost", action="store_true", help="Show estimated API cost for --model")
    parser.add_argument("--model", choices=list(PRICING.keys()), default="gpt4o",
                         help="Model used to estimate cost with --cost (default: gpt4o)")
    parser.add_argument("--compare", action="store_true", help="Show a side-by-side comparison table")
    args = parser.parse_args()

    print(f'Input: "{args.text}"  ({len(args.text)} characters)\n')

    results = [(name, fn(args.text)) for name, fn in TOKENIZERS]

    if args.compare:
        render_table(args.text, results)
    else:
        for name, tokens in results:
            ratio = compression_ratio(args.text, tokens)
            print(f"{name} — {len(tokens)} tokens — {ratio:.2f} chars/token")
            render_colored(tokens)
            print()

    if args.cost:
        n_tokens, cents = estimate_cost(args.text, args.model)
        print(f"Estimated cost on {args.model} ({n_tokens} tokens): {cents:.4f} cents (${cents / 100:.6f})")


if __name__ == "__main__":
    main()
