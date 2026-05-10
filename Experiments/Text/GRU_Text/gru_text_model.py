import torch
from torch import nn


class GRUTextDecoder(nn.Module):
    """
    GRU decoder for direct brain-to-text character decoding with CTC.
    """

    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=5,
        patch_size=0,
        patch_stride=0,
    ):
        """
        neural_dim   (int): number of neural features per timestep, e.g. 512
        n_units      (int): number of hidden units in each GRU layer
        n_days       (int): number of recording days / sessions
        n_classes    (int): number of output character classes for CTC
        rnn_dropout  (float): dropout between GRU layers
        input_dropout(float): dropout after day-specific input layer
        n_layers     (int): number of GRU layers
        patch_size   (int): number of timesteps to concatenate at input
        patch_stride (int): stride used in input patching
        """
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout

        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Day-specific affine input transforms
        self.day_layer_activation = nn.Softsign()

        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim

        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            batch_first=True,
            bidirectional=False,
        )

        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states=None, return_state=False):
        """
        x: tensor of shape [batch, time, neural_dim]
        day_idx: tensor with one day index per example in batch
        """
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        if self.patch_size > 0:
            x = x.unsqueeze(1)
            x = x.permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2)
            x_unfold = x_unfold.permute(0, 2, 3, 1)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        output, hidden_states = self.gru(x, states)
        logits = self.out(output)

        if return_state:
            return logits, hidden_states

        return logits