import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
batch_size = 64      # number of independent sequences processed in parallel
block_size = 256     # maximum context length (in tokens) used for predictions
max_iters = 5000     # total number of training iterations
eval_interval = 500  # how often we evaluate and print the loss
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200     # number of batches averaged when estimating loss
n_embd = 384         # embedding dimension (token + position vectors size)
n_head = 6           # number of attention heads per transformer block
n_layer = 6          # number of transformer blocks stacked
dropout = 0.2        # dropout probability (regularisation)
# ---------------------------------------------------------------------------

torch.manual_seed(1337)

# ---------------------------------------------------------------------------
# Data loading and tokenization
# ---------------------------------------------------------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Build a character-level vocabulary from the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mappings between characters and integer indices
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]          # string  -> list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # list of ints -> string

# Encode the full dataset and split into train (90%) / validation (10%)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def get_batch(split):
    """Return a random batch of (inputs, targets) from the chosen split.

    For each sampled starting index i, the input is data[i:i+block_size]
    and the target is data[i+1:i+block_size+1] — shifted by one position.
    Every token in the input therefore has a known next-token target.
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size]     for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ---------------------------------------------------------------------------
# Loss estimation
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    """Estimate mean loss over eval_iters batches for both splits.

    @torch.no_grad() disables gradient tracking to save memory and compute
    during evaluation. model.eval() / model.train() toggle behaviours like
    dropout that differ between training and inference.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class Head(nn.Module):
    """Single self-attention head.

    Each token produces a Query (what am I looking for?), a Key (what do I
    offer?) and a Value (what do I actually send if selected?). Attention
    scores are computed as scaled dot-products between queries and keys, then
    masked so that a token can only attend to earlier positions (causal mask).
    The output is a weighted sum of values according to those scores.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Causal (lower-triangular) mask stored as a non-trainable buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, _ = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)

        # Scaled dot-product attention scores
        # Dividing by sqrt(head_size) prevents the dot-products from growing
        # too large, which would push softmax into a near-zero-gradient region.
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)

        # Apply causal mask: future positions get -inf, which becomes 0 after softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Weighted aggregation of values
        v   = self.value(x)  # (B, T, hs)
        out = wei @ v         # (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple self-attention heads running in parallel.

    Each head learns to attend to different aspects of the context
    (e.g. syntax, semantics, positional patterns). Their outputs are
    concatenated and projected back to the embedding dimension.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from all heads along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """Position-wise feed-forward network applied independently to each token.

    Expands the representation to 4x the embedding size, applies a non-linearity
    (ReLU), then projects back. This lets each token "think" about the
    context it collected during attention.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: self-attention followed by feed-forward.

    Residual connections (x = x + sublayer(x)) let gradients flow directly
    through the network, enabling stable training of deep stacks. LayerNorm
    is applied before each sublayer (pre-norm formulation).
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # communication between tokens
        x = x + self.ffwd(self.ln2(x))  # per-token computation
        return x

# ---------------------------------------------------------------------------
# Model: GPT Language Model
# ---------------------------------------------------------------------------
class GPTLanguageModel(nn.Module):
    """Character-level GPT language model.

    Architecture:
      token embedding  ->  + position embedding
      -> n_layer transformer blocks
      -> final LayerNorm
      -> linear projection to vocab logits
    """

    def __init__(self):
        super().__init__()
        # Each token index maps to a learned vector of size n_embd
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        # Each position (0 to block_size-1) maps to a learned vector of size n_embd
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)  # final layer norm
        # Project from embedding space to vocabulary logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialise linear and embedding weights with a small normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token embeddings + position embeddings combined
        tok_emb = self.token_embedding_table(idx)                                # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb  # (B, T, C) — each token knows both its identity and its position

        x      = self.blocks(x)   # (B, T, C) — pass through all transformer blocks
        x      = self.ln_f(x)     # (B, T, C) — final normalisation
        logits = self.lm_head(x)  # (B, T, vocab_size) — raw scores for each next token

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Autoregressively append up to max_new_tokens tokens to the context.

        The context is cropped to block_size at each step because the position
        embedding table only has entries up to that length.
        """
        for _ in range(max_new_tokens):
            # Crop to the maximum supported context length
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits   = logits[:, -1, :]          # (B, C) — last time-step only
            probs    = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx      = torch.cat((idx, idx_next), dim=1)       # (B, T+1)
        return idx

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # Periodically evaluate and log loss on both splits
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch, forward pass, backprop, weight update
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
# Start from a single newline token (index 0) and let the model generate freely
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
