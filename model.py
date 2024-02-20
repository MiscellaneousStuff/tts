import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, head_size, n_embed, BLOCK_SIZE, dropout):
        super().__init__()
        self.k = nn.Linear(n_embed, head_size, bias=False)
        self.q = nn.Linear(n_embed, head_size, bias=False)
        self.v = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer("tril",
                             torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.k(x)
        q = self.q(x)
        w = q @ k.transpose(-2, -1) * C ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.v(x)
        o = w @ v
        return o

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads, head_size, dropout, BLOCK_SIZE):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embed, BLOCK_SIZE, dropout)
             for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        # print("MHSA proj.shape:", n_embed)
    def forward(self, x):
        # print("MHSA x.shape:", x.shape)
        o = torch.cat([h(x) for h in self.heads], dim=-1)
        # print("MHSA concat o.shape:", o.shape)
        o = self.dropout(self.proj(o))
        # print("MHSA project o.shape:", o.shape)
        return o
    
class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head, dropout, BLOCK_SIZE):
        super().__init__()
        head_size = n_embed // n_head
        self.sa   = MultiHeadAttention(n_embed, n_head, head_size, dropout, BLOCK_SIZE)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1  = nn.LayerNorm(n_embed)
        self.ln2  = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_len, n_embed, n_heads, n_layer, BLOCK_SIZE, dropout=0.2):
        super().__init__()
        self.BLOCK_SIZE = BLOCK_SIZE
        self.token_emb_table    = nn.Embedding(vocab_len, n_embed)
        self.position_emb_table = nn.Embedding(BLOCK_SIZE, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_heads, dropout=dropout, BLOCK_SIZE=BLOCK_SIZE)
              for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_len)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_embed = self.token_emb_table(idx)
        pos_embed = self.position_emb_table(
            torch.arange(T, device="cpu"))

        x = tok_embed + pos_embed
        x = self.blocks(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if not targets is None:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx