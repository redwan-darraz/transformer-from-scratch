"""
Terminal rendering: colored token view and side-by-side comparison table.
"""

from colorama import Back, Style, init as colorama_init

colorama_init(autoreset=True)

# Cycle of background colors used to tell consecutive tokens apart
TOKEN_COLORS = [Back.RED, Back.GREEN, Back.YELLOW, Back.BLUE, Back.MAGENTA, Back.CYAN]


def compression_ratio(text, tokens):
    """Characters per token — higher means the tokenizer compresses better."""
    if not tokens:
        return 0.0
    return len(text) / len(tokens)


def render_colored(tokens):
    """Print tokens with alternating background colors so you can see the split visually."""
    parts = []
    for i, token in enumerate(tokens):
        color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
        parts.append(f"{color}{token}{Style.RESET_ALL}")
    print("  " + "".join(parts))


def render_table(text, results):
    """Print a side-by-side table: Tokenizer | Tokens | Chars/token | Preview."""
    headers = ["Tokenizer", "Tokens", "Chars/token", "Preview"]
    rows = []
    for name, tokens in results:
        ratio = compression_ratio(text, tokens)
        preview = " ".join(tokens)
        if len(preview) > 40:
            preview = preview[:37] + "..."
        rows.append([name, str(len(tokens)), f"{ratio:.2f}", preview])

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(char_l, char_mid, char_r, fill="─"):
        return char_l + char_mid.join(fill * (w + 2) for w in widths) + char_r

    def row(cells):
        return "│ " + " │ ".join(c.ljust(w) for c, w in zip(cells, widths)) + " │"

    print(line("┌", "┬", "┐"))
    print(row(headers))
    print(line("├", "┼", "┤"))
    for r in rows:
        print(row(r))
    print(line("└", "┴", "┘"))
