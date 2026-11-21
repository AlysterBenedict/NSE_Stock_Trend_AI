"""PyTorch Global Panel TemporalAttentionNet with Stock & Sector Entity Embeddings + FiLM Conditioning.

Designed for native NVIDIA RTX 4060 CUDA (AMP FP16/TF32) training across ~150,000+ NSE panel sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_torch_device():
    """Return CUDA device when an NVIDIA GPU is available and enable Tensor Core optimizations."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


class CausalDilatedConvBlock(nn.Module):
    """Causal 1D dilated convolution residual block with LayerNorm and GELU."""

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.25):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=self.padding
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=self.padding
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = self.conv1(x.transpose(1, 2))
        if self.padding > 0:
            h = h[:, :, : -self.padding]
        h = self.dropout(F.gelu(self.norm1(h.transpose(1, 2))))

        h = self.conv2(h.transpose(1, 2))
        if self.padding > 0:
            h = h[:, :, : -self.padding]
        h = self.dropout(F.gelu(self.norm2(h.transpose(1, 2))))
        return residual + h


class TemporalAttentionNet(nn.Module):
    """Global Cross-Sectional TCN + BiGRU + Multi-Head Attention + Entity FiLM Conditioning.

    Shares temporal representation weights across all 58 NSE stocks while specializing per-stock
    and per-sector dynamics through learned entity embeddings + Feature-wise Linear Modulation (FiLM).
    """

    def __init__(
        self,
        input_dim,
        num_stocks=64,
        num_sectors=16,
        d_model=128,
        gru_hidden=96,
        num_heads=4,
        dropout=0.30,
        noise_std=0.02,
    ):
        super().__init__()
        self.config = {
            "input_dim": int(input_dim),
            "num_stocks": int(num_stocks),
            "num_sectors": int(num_sectors),
            "d_model": int(d_model),
            "gru_hidden": int(gru_hidden),
            "num_heads": int(num_heads),
            "dropout": float(dropout),
            "noise_std": float(noise_std),
        }
        self.noise_std = float(noise_std)

        self.stock_emb = nn.Embedding(num_stocks + 1, 16)
        self.sector_emb = nn.Embedding(num_sectors + 1, 8)
        entity_dim = 24

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # FiLM conditioning: stock/sector identity modulates hidden channel scale & shift
        self.film = nn.Linear(entity_dim, d_model * 2)

        self.tcn = nn.Sequential(
            CausalDilatedConvBlock(d_model, kernel_size=3, dilation=1, dropout=dropout),
            CausalDilatedConvBlock(d_model, kernel_size=3, dilation=2, dropout=dropout),
            CausalDilatedConvBlock(d_model, kernel_size=3, dilation=4, dropout=dropout),
        )

        self.bigru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        attn_dim = gru_hidden * 2

        self.attn_norm = nn.LayerNorm(attn_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(attn_dim)
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 2, attn_dim),
            nn.Dropout(dropout),
        )

        self.pool_weights = nn.Sequential(
            nn.Linear(attn_dim, attn_dim // 2),
            nn.Tanh(),
            nn.Linear(attn_dim // 2, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(attn_dim * 2 + entity_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, stock_id=None, sector_id=None):
        # x: (B, T, input_dim)
        bsz = x.size(0)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        if stock_id is None:
            stock_id = torch.full((bsz,), self.config["num_stocks"], dtype=torch.long, device=x.device)
        if sector_id is None:
            sector_id = torch.full((bsz,), self.config["num_sectors"], dtype=torch.long, device=x.device)

        stock_id = stock_id.clamp(0, self.config["num_stocks"])
        sector_id = sector_id.clamp(0, self.config["num_sectors"])
        entity_vec = torch.cat([self.stock_emb(stock_id), self.sector_emb(sector_id)], dim=-1)

        h = self.input_proj(x)
        gamma, beta = self.film(entity_vec).chunk(2, dim=-1)
        h = h * (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(1)) + 0.1 * beta.unsqueeze(1)

        h = self.tcn(h)
        h, _ = self.bigru(h)

        h_norm = self.attn_norm(h)
        attn_out, _ = self.self_attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + attn_out
        h = h + self.ffn(self.ffn_norm(h))

        weights = torch.softmax(self.pool_weights(h), dim=1)
        pooled = torch.sum(h * weights, dim=1)
        last_step = h[:, -1, :]

        fused = torch.cat([pooled, last_step, entity_vec], dim=-1)
        return self.head(fused).squeeze(-1)


def save_attention_net_bundle(models, filepath):
    """Save a single model or multi-seed bagged list of TemporalAttentionNet checkpoints."""
    model_list = models if isinstance(models, list) else [models]
    payload = {
        "config": model_list[0].config,
        "state_dicts": [
            {k: v.detach().cpu() for k, v in m.state_dict().items()} for m in model_list
        ],
    }
    torch.save(payload, filepath)


def load_attention_net_bundle(filepath, device=None):
    """Load a persisted single or multi-seed bagged list of TemporalAttentionNet models in eval mode."""
    target_device = device if device is not None else get_torch_device()
    checkpoint = torch.load(filepath, map_location=target_device, weights_only=True)
    state_dicts = checkpoint.get("state_dicts") or [checkpoint["state_dict"]]
    models = []
    for sd in state_dicts:
        model = TemporalAttentionNet(**checkpoint["config"])
        model.load_state_dict(sd)
        model.to(target_device)
        model.eval()
        models.append(model)
    return models
