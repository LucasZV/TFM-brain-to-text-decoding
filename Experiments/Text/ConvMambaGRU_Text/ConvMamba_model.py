import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0, mlp_ratio=2.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout1 = nn.Dropout(dropout)

        mlp_hidden = int(d_model * mlp_ratio)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.mamba(self.norm1(x)))
        x = x + self.dropout2(self.mlp(self.norm2(x)))
        return x


class ConvFrontend(nn.Module):
    """
    Front-end convolucional temporal ligero.
    Entrada: [B, T, C]
    Salida:  [B, T, C]
    """

    def __init__(self, channels, dropout=0.0, kernel_size=5):
        super().__init__()

        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.GELU(),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),

            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.GELU(),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        residual = x
        x = self.net(x)
        x = x + residual
        x = x.transpose(1, 2)
        return x


class MambaTextDecoder(nn.Module):
    """
    Conv + Mamba + GRU final para decodificación directa a caracteres con CTC.

    Mantiene:
    - capas específicas por día
    - front-end convolucional temporal
    - patching temporal
    - encoder Mamba
    - refinamiento GRU al final
    - salida a clases
    """

    def __init__(
        self,
        neural_dim,
        n_classes,
        n_days,
        hidden_dim=768,
        n_layers=5,
        patch_size=14,
        patch_stride=4,
        input_dropout=0.0,
        dropout=0.0,
        d_state=16,
        d_conv=4,
        expand=2,
        conv_kernel_size=5,
        conv_dropout=0.1,
        mlp_ratio=2.0,
        gru_hidden_dim=None,
        gru_layers=1,
        gru_dropout=0.0,
    ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.n_days = n_days
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.gru_hidden_dim = hidden_dim if gru_hidden_dim is None else gru_hidden_dim
        self.gru_layers = gru_layers

        # Day-specific adaptation
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        # Conv front-end
        self.conv_frontend = ConvFrontend(
            channels=neural_dim,
            dropout=conv_dropout,
            kernel_size=conv_kernel_size,
        )

        # Patching + projection
        input_size = neural_dim
        if self.patch_size > 0:
            input_size = neural_dim * patch_size

        self.input_projection = nn.Linear(input_size, hidden_dim)

        # Mamba encoder
        self.mamba_blocks = nn.ModuleList([
            MambaResidualBlock(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(n_layers)
        ])

        # GRU final de refinamiento
        effective_gru_dropout = gru_dropout if gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=effective_gru_dropout,
            bidirectional=False,
        )

        self.final_norm = nn.LayerNorm(self.gru_hidden_dim)
        self.final_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.gru_hidden_dim, n_classes)

    def _apply_day_layer(self, x, day_idx):
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        x = self.day_layer_dropout(x)
        return x

    def _apply_patching(self, x):
        if self.patch_size <= 0:
            return x

        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        return x

    def forward(self, x, day_idx):
        x = self._apply_day_layer(x, day_idx)
        x = self.conv_frontend(x)
        x = self._apply_patching(x)
        x = self.input_projection(x)

        for block in self.mamba_blocks:
            x = block(x)

        x, _ = self.gru(x)
        x = self.final_norm(x)
        x = self.final_dropout(x)

        logits = self.out(x)
        return logits