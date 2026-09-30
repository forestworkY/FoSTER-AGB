# GEDI-T: GEDI structural baseline.
# Concat-T: fixed GEDI–Landsat fusion.
# State-T: FT-Transformer forest-state representation.
# FoSTER: FT-Transformer with state-dependent residual expert routing.
# FoSTER-SC: FoSTER with state-conditioned residual correction.

import torch
import torch.nn as nn

from .comparison_models import M3


class FoSTER(nn.Module):
    def __init__(self, in_dim_G, in_dim_S, in_dim_E, n_experts=4, hidden=64, n_pred_bins=8, n_residual_bins=6, max_residual=0.25, max_calibration=0.30, dropout=0.15):
        super().__init__()
        self.backbone = M3(in_dim_G, in_dim_S, in_dim_E)
        d = self.backbone.norm.normalized_shape[0]
        self.n_pred_bins = n_pred_bins
        self.max_residual = max_residual
        self.max_calibration = max_calibration
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.reference_calibration = nn.Sequential(
            nn.Linear(d + 1, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1), nn.Tanh()
        )
        self.pred_bin_embedding = nn.Embedding(n_pred_bins, n_pred_bins)
        self.router = nn.Sequential(
            nn.Linear(3 + 5 + n_pred_bins, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_experts)
        )
        expert_dim = 4 * d + 1
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_dim, hidden), nn.LayerNorm(hidden),
                          nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1), nn.Tanh())
            for _ in range(n_experts - 1)
        ])
        self.residual_state_head = nn.Sequential(
            nn.Linear(d + 4, hidden), nn.GELU(), nn.Linear(hidden, n_residual_bins)
        )

    def forward(self, x_G, x_S, x_E, uncertainty, prediction_bin_edges=None):
        if uncertainty.ndim != 2 or uncertainty.shape != (x_G.shape[0], 5):
            raise ValueError("uncertainty must have shape (batch_size, 5)")
        y_base, z_G, z_S, z_E, z_c, alpha = self.backbone(x_G, x_S, x_E)
        state_calibration = self.max_calibration * self.reference_calibration(torch.cat((z_c, y_base), dim=-1))
        reference = y_base + self.global_bias + state_calibration
        if prediction_bin_edges is None:
            bin_embedding = x_G.new_zeros((x_G.shape[0], self.n_pred_bins))
        else:
            edges = torch.as_tensor(prediction_bin_edges, dtype=y_base.dtype, device=y_base.device)
            if edges.numel() != self.n_pred_bins + 1:
                raise ValueError("prediction_bin_edges must have n_pred_bins + 1 values")
            raw_base = torch.expm1(y_base.detach().clamp(-1.0, 8.0))
            bin_index = torch.bucketize(raw_base.squeeze(-1), edges[1:-1]).clamp(0, self.n_pred_bins - 1)
            bin_embedding = self.pred_bin_embedding(bin_index)
        router_input = torch.cat((alpha.detach(), uncertainty, bin_embedding), dim=-1)
        expert_weights = torch.softmax(self.router(router_input), dim=-1)
        expert_input = torch.cat((z_G, z_S, z_E, z_c, y_base), dim=-1)
        corrections = [y_base.new_zeros(y_base.shape)]
        corrections.extend(self.max_residual * expert(expert_input) for expert in self.experts)
        residual = (expert_weights * torch.cat(corrections, dim=-1)).sum(dim=-1, keepdim=True)
        residual_logits = self.residual_state_head(torch.cat((z_c, state_calibration, alpha), dim=-1))
        return {
            "y_log": reference + residual,
            "reference_log": reference,
            "base_log": y_base,
            "residual_log": residual,
            "expert_weights": expert_weights,
            "source_attention": alpha,
            "forest_state": z_c,
            "residual_logits": residual_logits,
        }
