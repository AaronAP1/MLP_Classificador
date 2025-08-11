import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, cat_features, emb_dim):
        super().__init__()
        self.num_features = num_features
        self.cat_features = cat_features
        self.emb_dim = emb_dim
        self.num_linear = nn.ModuleList([nn.Linear(1, emb_dim) for _ in range(num_features)])
        self.cat_emb = nn.ModuleList([nn.Embedding(cat_size, emb_dim) for cat_size in cat_features])
    def forward(self, x_num, x_cat):
        num_tokens = [layer(x_num[:,i:i+1]) for i, layer in enumerate(self.num_linear)]
        cat_tokens = [emb(x_cat[:,i]) for i, emb in enumerate(self.cat_emb)]
        tokens = torch.stack(num_tokens + cat_tokens, dim=1)
        return tokens

class FTTransformer(nn.Module):
    def __init__(self, num_features, cat_features, emb_dim, n_heads, n_layers, num_classes, dropout=0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, cat_features, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
    def forward(self, x_num, x_cat):
        B = x_num.size(0)
        tokens = self.tokenizer(x_num, x_cat)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)
        cls_out = self.norm(out[:,0])
        logits = self.head(cls_out)
        return logits, out[:,0]  # logits, cls embedding
