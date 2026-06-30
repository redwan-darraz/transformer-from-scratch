"""
tok — compare BPE from-scratch, tiktoken and HuggingFace on any text.
Usage: python -m tok "your text here" [--verbose] [--stats]
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="tok",
        description="Tokenize text and compare BPE, tiktoken and HuggingFace side by side.",
    )
    parser.add_argument("text", help="Text to tokenize")
    parser.add_argument("--verbose", action="store_true", help="Show full token list per tokenizer")
    parser.add_argument("--stats", action="store_true", help="Show compression stats only")
    args = parser.parse_args()

    print(f"Input: \"{args.text}\"")
    print()
    print("Coming soon — tokenizers will be implemented here.")


if __name__ == "__main__":
    main()
