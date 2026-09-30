import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .config import ModelConfig, ALL_FEATURE_COLS

CFG = ModelConfig()


class ResBlock(nn.Module):

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ResMLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, n_blocks=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.head(self.blocks(self.proj(x)))


class PredHead(nn.Module):

    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):

    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        xn = self.norm1(x)
        attn_out, attn_w = self.attn(xn, xn, xn, need_weights=return_attn, average_attn_weights=True)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return (x, attn_w) if return_attn else x


class M1(nn.Module):

    def __init__(self, in_dim_G):
        super().__init__()
        self.encoder_G = ResMLP(in_dim_G, CFG.hidden_dim, CFG.z_dim, 2, CFG.dropout)
        self.pred_head = PredHead(CFG.z_dim, CFG.hidden_dim)

    def forward(self, x_G):
        z_G = self.encoder_G(x_G)
        return (self.pred_head(z_G), z_G)


class M2(nn.Module):

    def __init__(self, in_dim_G, in_dim_S, in_dim_E):
        super().__init__()
        self.encoder_G = ResMLP(in_dim_G, CFG.hidden_dim, CFG.z_dim, 2, CFG.dropout)
        self.encoder_S = ResMLP(in_dim_S, CFG.hidden_dim, CFG.z_dim, 2, CFG.dropout)
        self.encoder_E = ResMLP(in_dim_E, CFG.hidden_dim, CFG.z_dim, 2, CFG.dropout)
        self.pred_head = PredHead(CFG.z_dim * 3, CFG.hidden_dim)

    def forward(self, x_G, x_S, x_E):
        z_G = self.encoder_G(x_G)
        z_S = self.encoder_S(x_S)
        z_E = self.encoder_E(x_E)
        return (self.pred_head(torch.cat([z_G, z_S, z_E], dim=-1)), z_G, z_S, z_E)


class M3(nn.Module):

    def __init__(self, in_dim_G: int, in_dim_S: int, in_dim_E: int):
        super().__init__()
        d = CFG.t_d_model
        nh = CFG.t_n_heads
        nl = CFG.t_n_layers
        fd = d * CFG.t_ffn_mult
        dr = CFG.dropout
        self.in_dim_G = in_dim_G
        self.in_dim_S = in_dim_S
        self.in_dim_E = in_dim_E
        n_feat = in_dim_G + in_dim_S + in_dim_E
        self.W_feat = nn.Parameter(torch.empty(n_feat, d))
        self.b_feat = nn.Parameter(torch.zeros(n_feat, d))
        nn.init.normal_(self.W_feat, std=0.01)
        self.mod_embed = nn.Embedding(3, d)
        mod_ids = torch.cat([torch.zeros(in_dim_G, dtype=torch.long), torch.ones(in_dim_S, dtype=torch.long), torch.full((in_dim_E,), 2, dtype=torch.long)])
        self.register_buffer('mod_ids', mod_ids)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.layers = nn.ModuleList([TransformerLayer(d, nh, fd, dr) for _ in range(nl)])
        self.norm = nn.LayerNorm(d)
        self.pred_head = PredHead(d, CFG.hidden_dim)
        self.state_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, CFG.hidden_dim), nn.GELU(), nn.Dropout(dr), nn.Linear(CFG.hidden_dim, len(CFG.m3_state_summary_cols)))

    def forward(self, x_G, x_S, x_E, return_state=False):
        B = x_G.shape[0]
        d = CFG.t_d_model
        x_all = torch.cat([x_G, x_S, x_E], dim=-1)
        tokens = x_all.unsqueeze(-1) * self.W_feat.unsqueeze(0) + self.b_feat.unsqueeze(0)
        tokens = tokens + self.mod_embed(self.mod_ids).unsqueeze(0)
        cls = self.cls_token.expand(B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)
        for i, layer in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            if is_last:
                tokens, attn_w = layer(tokens, return_attn=True)
            else:
                tokens = layer(tokens)
        tokens = self.norm(tokens)
        z_c = tokens[:, 0]
        feat = tokens[:, 1:]
        z_G = feat[:, :self.in_dim_G].mean(dim=1)
        z_S = feat[:, self.in_dim_G:self.in_dim_G + self.in_dim_S].mean(dim=1)
        z_E = feat[:, self.in_dim_G + self.in_dim_S:].mean(dim=1)
        cls_attn = attn_w[:, 0, 1:]
        ag = cls_attn[:, :self.in_dim_G].sum(dim=1, keepdim=True)
        as_ = cls_attn[:, self.in_dim_G:self.in_dim_G + self.in_dim_S].sum(dim=1, keepdim=True)
        ae = cls_attn[:, self.in_dim_G + self.in_dim_S:].sum(dim=1, keepdim=True)
        alpha = torch.cat([ag, as_, ae], dim=-1)
        y_hat = self.pred_head(z_c)
        if return_state:
            state_hat = self.state_head(z_c)
            return (y_hat, z_G, z_S, z_E, z_c, alpha, state_hat)
        return (y_hat, z_G, z_S, z_E, z_c, alpha)


def is_m3(model_name: str) -> bool:
    return model_name.upper() == 'M3'


def get_m3_state_indices():
    return [ALL_FEATURE_COLS.index(c) for c in CFG.m3_state_summary_cols if c in ALL_FEATURE_COLS]


def set_m3_learning_rate(optimizer, epoch: int):
    warmup = max(0, int(CFG.m3_warmup_epochs))
    eta_min_ratio = CFG.m3_eta_min / CFG.m3_lr
    if warmup > 0 and epoch <= warmup:
        scale = epoch / warmup
    else:
        denom = max(1, CFG.max_epochs - warmup)
        progress = min(1.0, max(0.0, (epoch - warmup) / denom))
        scale = eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group['lr'] = CFG.m3_lr * scale


def forward_m3_with_state(model, batch_x, idx_G, idx_S, idx_E):
    return model(batch_x[:, idx_G], batch_x[:, idx_S], batch_x[:, idx_E], return_state=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = (y_true[mask], y_pred[mask])
    n = len(y_t)
    if n == 0:
        return {'n': 0, 'R2': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Bias': np.nan, 'nRMSE': np.nan}
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    return {'n': n, 'R2': r2_score(y_t, y_p), 'RMSE': rmse, 'MAE': mean_absolute_error(y_t, y_p), 'Bias': float(np.mean(y_p - y_t)), 'nRMSE': rmse / (np.mean(y_t) + 1e-06)}


def compute_high_agb_metrics(y_true, y_pred, threshold_q=0.8) -> dict:
    thr = np.quantile(y_true, threshold_q)
    mask = y_true >= thr
    if mask.sum() < 5:
        return {'n_high': 0, 'R2_high': np.nan, 'RMSE_high': np.nan, 'Bias_high': np.nan}
    m = compute_metrics(y_true[mask], y_pred[mask])
    return {'n_high': m['n'], 'R2_high': m['R2'], 'RMSE_high': m['RMSE'], 'Bias_high': m['Bias']}


def forward_batch(model, batch_x, model_name, idx_G, idx_S, idx_E, idx_S_robust=None):
    name = model_name.upper()
    x_G = batch_x[:, idx_G]
    if name == 'M1':
        return model(x_G)[0]
    if name in ('M2', 'M3'):
        return model(x_G, batch_x[:, idx_S], batch_x[:, idx_E])[0]
    raise ValueError(name)


class EarlyStopping:

    def __init__(self, patience: int=20, min_delta: float=1e-05):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_fold(model, train_loader, val_loader, model_name: str, idx_G, idx_S, idx_E, device: torch.device, verbose: bool=True, idx_S_robust=None):
    m3 = is_m3(model_name)
    lr = CFG.m3_lr if m3 else CFG.lr
    weight_decay = CFG.m3_weight_decay if m3 else CFG.weight_decay
    patience = CFG.m3_patience if m3 else CFG.patience
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None if m3 else CosineAnnealingLR(optimizer, T_max=CFG.max_epochs, eta_min=1e-05)
    criterion = nn.MSELoss()
    state_indices = get_m3_state_indices() if m3 and CFG.m3_state_aux_weight > 0 else []
    state_criterion = nn.MSELoss()
    stopper = EarlyStopping(patience=patience)
    t0 = time.time()
    for epoch in range(1, CFG.max_epochs + 1):
        if m3:
            set_m3_learning_rate(optimizer, epoch)
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = (X_batch.to(device), y_batch.to(device))
            optimizer.zero_grad()
            if m3 and state_indices:
                out = forward_m3_with_state(model, X_batch, idx_G, idx_S, idx_E)
                y_hat, state_hat = (out[0], out[-1])
                state_target = X_batch[:, state_indices]
                loss = criterion(y_hat, y_batch) + CFG.m3_state_aux_weight * state_criterion(state_hat, state_target)
            else:
                y_hat = forward_batch(model, X_batch, model_name, idx_G, idx_S, idx_E, idx_S_robust)
                loss = criterion(y_hat, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = (X_batch.to(device), y_batch.to(device))
                y_hat = forward_batch(model, X_batch, model_name, idx_G, idx_S, idx_E, idx_S_robust)
                val_loss += criterion(y_hat, y_batch).item() * len(y_batch)
        val_loss /= len(val_loader.dataset)
        if m3:
            set_m3_learning_rate(optimizer, epoch + 1)
        else:
            scheduler.step()
        if verbose and epoch % 20 == 0:
            elapsed = time.time() - t0
            print(f'  ep {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:.0f}s')
        if stopper.step(val_loss, model):
            if verbose:
                print(f'  Early stop at epoch {epoch}, best val={stopper.best_loss:.4f}')
            break
    stopper.restore(model)
    return stopper.best_loss


@torch.no_grad()
def predict(model, X_test: np.ndarray, model_name: str, idx_G, idx_S, idx_E, device: torch.device, batch_size: int=1024, idx_S_robust=None):
    model.eval()
    X_t = torch.from_numpy(X_test).to(device)
    preds = []
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i:i + batch_size]
        out = forward_batch(model, batch, model_name, idx_G, idx_S, idx_E, idx_S_robust)
        preds.append(out.cpu().numpy())
    log_pred = np.concatenate(preds).squeeze()
    log_pred = np.clip(log_pred, -1.0, 8.0)
    raw_pred = np.expm1(log_pred)
    raw_pred = np.clip(raw_pred, 0, 3000)
    return (log_pred, raw_pred)


@torch.no_grad()
def get_representations(model, X_test: np.ndarray, model_name: str, idx_G, idx_S, idx_E, device: torch.device, batch_size: int=1024):
    if model_name.upper() != 'M3':
        return (None, None)
    model.eval()
    X_t = torch.from_numpy(X_test).to(device)
    zcs, alphas = ([], [])
    for i in range(0, len(X_t), batch_size):
        b = X_t[i:i + batch_size]
        out = model(b[:, idx_G], b[:, idx_S], b[:, idx_E])
        zcs.append(out[4].cpu().numpy())
        alphas.append(out[5].cpu().numpy())
    return (np.concatenate(zcs), np.concatenate(alphas))
