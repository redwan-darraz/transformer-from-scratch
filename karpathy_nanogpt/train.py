import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
batch_size = 32      # number of independent sequences processed in parallel
block_size = 8       # maximum context length (in tokens) used for predictions
max_iters = 3000     # total number of training iterations
eval_interval = 300  # how often we evaluate and print the loss
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200     # number of batches averaged when estimating loss
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
# Model: Bigram Language Model (baseline)
# ---------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    """Simplest possible language model: predict the next token from the current
    token only, using a direct lookup table (embedding matrix of shape
    vocab_size x vocab_size). Each row stores the raw logits for the next token
    given the corresponding input token.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx:    (B, T) integer token indices
        # logits: (B, T, C) raw scores for each next token, C = vocab_size
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # Flatten to (B*T, C) and (B*T,) as required by F.cross_entropy
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Autoregressively append up to max_new_tokens tokens to the context.

        At each step:
          1. Forward pass to get logits for the whole sequence.
          2. Keep only the last time-step logits (next-token prediction).
          3. Convert to probabilities via softmax.
          4. Sample one token from that distribution.
          5. Append the token to the running context and repeat.
        """
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits   = logits[:, -1, :]          # (B, C) — last time-step only
            probs    = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx      = torch.cat((idx, idx_next), dim=1)       # (B, T+1)
        return idx

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # Periodically evaluate and log loss on both splits
    if iter % eval_interval == 0:
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
