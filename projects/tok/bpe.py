"""
Minimal BPE tokenizer, trained on a small English corpus at import time.
Same algorithm as lab01_tokenizer, extracted here as a reusable module.
"""

from collections import defaultdict

CORPUS = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "a quick brown fox jumps over the lazy dog",
    "the model learns from training data",
    "attention allows the model to focus on relevant tokens",
    "the embedding layer maps tokens to dense vectors",
    "gradient descent minimizes the loss function",
    "the transformer architecture uses self attention",
    "tokenization splits text into smaller units called tokens",
    "the vocabulary contains all possible tokens",
    "byte pair encoding merges the most frequent pairs",
    "the tokenizer encodes text as a list of integers",
    "machine learning models improve with more data",
    "neural networks are inspired by the brain",
    "the language model predicts the next token",
    "natural language processing deals with human language",
]

NUM_MERGES = 80


def _build_vocab(corpus):
    vocab = defaultdict(int)
    for sentence in corpus:
        for word in sentence.split():
            vocab[tuple(word) + ('</w>',)] += 1
    return dict(vocab)


def _get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return dict(pairs)


def _merge_vocab(pair, vocab):
    new_vocab = {}
    merged_symbol = pair[0] + pair[1]
    for word, freq in vocab.items():
        new_word, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged_symbol)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


def _apply_merges(symbols, merges):
    for pair, merged_symbol in merges:
        i, new_symbols = 0, []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(merged_symbol)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols


def _train():
    vocab = _build_vocab(CORPUS)
    merges = []
    for _ in range(NUM_MERGES):
        stats = _get_stats(vocab)
        if not stats:
            break
        best_pair = max(stats, key=stats.get)
        merges.append((best_pair, best_pair[0] + best_pair[1]))
        vocab = _merge_vocab(best_pair, vocab)
    return merges


MERGES = _train()


def tokenize(text):
    """Return the list of BPE token strings for the given text (lowercased)."""
    tokens = []
    for word in text.lower().split():
        symbols = _apply_merges(list(word) + ['</w>'], MERGES)
        tokens.extend(s.replace('</w>', '') for s in symbols)
    return [t for t in tokens if t]
