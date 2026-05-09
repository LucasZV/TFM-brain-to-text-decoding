import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaDecoder(nn.Module):
    """
    Drop-in replacement inicial del GRU baseline:
    - capas de entrada específicas por día
    - patching temporal
    - encoder Mamba
    - capa de salida a clases fonémicas
    """

    def __init__(
        self,
        neural_dim,
        n_classes,
        n_days,
        hidden_dim=768,
        n_layers=6,
        patch_size=14,
        patch_stride=4,
        input_dropout=0.0,
        dropout=0.0,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.n_days = n_days
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        input_size = neural_dim
        if self.patch_size > 0:
            input_size = neural_dim * patch_size

        self.input_projection = nn.Linear(input_size, hidden_dim)

        self.mamba_blocks = nn.ModuleList([
            MambaResidualBlock(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, n_classes)

    def _apply_day_layer(self, x, day_idx):
        # x: [B, T, D]
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)  # [B, D, D]
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)  # [B, 1, D]

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        x = self.day_layer_dropout(x)
        return x

    def _apply_patching(self, x):
        # x: [B, T, D]
        if self.patch_size <= 0:
            return x

        # unfold sobre dimensión temporal
        # salida: [B, T_out, D, patch_size]
        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)

        # [B, T_out, patch_size, D]
        x = x.permute(0, 1, 3, 2).contiguous()

        # [B, T_out, patch_size * D]
        x = x.view(x.shape[0], x.shape[1], -1)
        return x

    def forward(self, x, day_idx, states=None, return_state=False):
        x = self._apply_day_layer(x, day_idx)
        x = self._apply_patching(x)
        x = self.input_projection(x)

        for block in self.mamba_blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.out(x)

        if return_state:
            return logits, None
        return logits
