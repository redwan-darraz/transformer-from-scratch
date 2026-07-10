"""
Run all three tokenizers on a batch of sentences and plot a token-count comparison.
Usage: python benchmark.py
"""

import matplotlib.pyplot as plt
import numpy as np

from engines import TOKENIZERS

SENTENCES = [
    "hello world",
    "the model learns from training data",
    "gradient descent minimizes the loss",
    "backpropagation",
    "GPT-4 was released in 2023!",
]


def main():
    results = {name: [] for name, _ in TOKENIZERS}
    for sentence in SENTENCES:
        print(f'"{sentence}"')
        for name, fn in TOKENIZERS:
            n = len(fn(sentence))
            results[name].append(n)
            print(f"  {name:<24} {n} tokens")
        print()

    x = np.arange(len(SENTENCES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, counts) in enumerate(results.items()):
        bars = ax.bar(x + i * width, counts, width, label=name)
        ax.bar_label(bars, fontsize=8, padding=2)

    ax.set_xticks(x + width)
    ax.set_xticklabels([s if len(s) < 25 else s[:22] + "..." for s in SENTENCES],
                        rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Number of tokens")
    ax.set_title("Token count per sentence — BPE vs tiktoken vs HuggingFace (Mistral)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("tokenizer_comparison.png", dpi=120)
    print("Saved: tokenizer_comparison.png")


if __name__ == "__main__":
    main()
