import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------
vocab_size = 65     # E.g., character-level vocabulary
n_embd = 64         # Embedding dimension
n_head = 4          # Number of attention heads (head_dim = 64 / 4 = 16)
n_layer = 3         # Number of Transformer blocks
block_size = 32     # Maximum context length
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------
# 1. Causal Self-Attention
# ---------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert n_embd % n_head == 0
        self.head_dim = n_embd // n_head
        
        # Combined Q, K, V projection
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Lower-triangular causal mask: tokens can only attend to past and current tokens
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # Batch, Sequence Length, Embedding Dim

        # Compute Q, K, V and split into heads
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Aggregate values and project output
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# ---------------------------------------------------------
# 2. Feed-Forward Network (MLP)
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard GPT expands the embedding dim by 4x in the hidden layer
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# ---------------------------------------------------------
# 3. Transformer Block
# ---------------------------------------------------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        # Pre-layer normalization with residual stream connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------
# 4. Mini-LLM Core Architecture
# ---------------------------------------------------------
class MiniLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Tie weights: share token embedding and output projection matrix
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Combine token and positional representations
        x = self.token_embedding(idx) + self.position_embedding(pos)

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # Project back to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten across batch and sequence length to calculate cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive text generation."""
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = self(idx_cond)
            
            # Focus on the last token's predictions
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


# ---------------------------------------------------------
# 5. Quick Test Run
# ---------------------------------------------------------
if __name__ == "__main__":
    model = MiniLLM().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy inputs: Batch of 2, Sequence length of 8 tokens
    dummy_x = torch.randint(0, vocab_size, (2, 8)).to(device)
    dummy_targets = torch.randint(0, vocab_size, (2, 8)).to(device)

    # Forward pass
    logits, loss = model(dummy_x, dummy_targets)
    print(f"Logits shape: {logits.shape}")  # (2, 8, 65)
    print(f"Sample Loss:  {loss.item():.4f}")

    # Generate next 10 tokens from an initial start token
    start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(start_context, max_new_tokens=10)
    print(f"Generated token sequence: {generated.tolist()[0]}")