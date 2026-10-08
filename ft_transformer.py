import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        normalized = self.norm1(x)
        attended, weights = self.attn(normalized, normalized, normalized, need_weights=return_attn, average_attn_weights=True)
        x = x + self.drop(attended)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return (x, weights) if return_attn else x


class PredHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(x)


class FTTransformerBackbone(nn.Module):
    def __init__(self, in_dim_G, in_dim_S, in_dim_E, d_model=64, n_heads=4, n_layers=2, ffn_mult=2, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.in_dim_G = in_dim_G
        self.in_dim_S = in_dim_S
        self.in_dim_E = in_dim_E
        self.d_model = d_model
        n_features = in_dim_G + in_dim_S + in_dim_E
        self.W_feat = nn.Parameter(torch.empty(n_features, d_model))
        self.b_feat = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.normal_(self.W_feat, std=0.01)
        self.mod_embed = nn.Embedding(3, d_model)
        mod_ids = torch.cat((
            torch.zeros(in_dim_G, dtype=torch.long),
            torch.ones(in_dim_S, dtype=torch.long),
            torch.full((in_dim_E,), 2, dtype=torch.long),
        ))
        self.register_buffer("mod_ids", mod_ids)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_model * ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pred_head = PredHead(d_model, hidden_dim)

    def forward(self, x_G, x_S, x_E):
        batch_size = x_G.shape[0]
        x_all = torch.cat((x_G, x_S, x_E), dim=-1)
        tokens = x_all.unsqueeze(-1) * self.W_feat.unsqueeze(0) + self.b_feat.unsqueeze(0)
        tokens = tokens + self.mod_embed(self.mod_ids).unsqueeze(0)
        cls = self.cls_token.expand(batch_size, 1, self.d_model)
        tokens = torch.cat((cls, tokens), dim=1)
        for index, layer in enumerate(self.layers):
            if index == len(self.layers) - 1:
                tokens, attention = layer(tokens, return_attn=True)
            else:
                tokens = layer(tokens)
        tokens = self.norm(tokens)
        z_c = tokens[:, 0]
        features = tokens[:, 1:]
        z_G = features[:, :self.in_dim_G].mean(dim=1)
        z_S = features[:, self.in_dim_G:self.in_dim_G + self.in_dim_S].mean(dim=1)
        z_E = features[:, self.in_dim_G + self.in_dim_S:].mean(dim=1)
        cls_attention = attention[:, 0, 1:]
        alpha_G = cls_attention[:, :self.in_dim_G].sum(dim=1, keepdim=True)
        alpha_S = cls_attention[:, self.in_dim_G:self.in_dim_G + self.in_dim_S].sum(dim=1, keepdim=True)
        alpha_E = cls_attention[:, self.in_dim_G + self.in_dim_S:].sum(dim=1, keepdim=True)
        alpha = torch.cat((alpha_G, alpha_S, alpha_E), dim=-1)
        return self.pred_head(z_c), z_G, z_S, z_E, z_c, alpha
