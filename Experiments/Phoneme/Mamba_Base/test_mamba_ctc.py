import torch
import torch.nn as nn
from mamba_model import BrainToTextMamba

device = "cuda" if torch.cuda.is_available() else "cpu"

B, T, C = 4, 320, 512
V = 35

x = torch.randn(B, T, C, device=device)
input_lengths = torch.tensor([320, 300, 280, 260], dtype=torch.long)
target_lengths = torch.tensor([20, 18, 16, 15], dtype=torch.long)

targets = torch.randint(low=1, high=V, size=(target_lengths.sum().item(),), dtype=torch.long)

model = BrainToTextMamba(
    input_dim=C,
    d_model=256,
    num_layers=6,
    vocab_size=V,
).to(device)

logits, _ = model(x, input_lengths.to(device))
log_probs = logits.log_softmax(-1).transpose(0, 1).cpu()  # [T, B, V]

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
loss = criterion(log_probs, targets, input_lengths, target_lengths)

print("CTC loss:", float(loss))
