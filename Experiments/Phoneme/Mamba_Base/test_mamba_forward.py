import torch
from mamba_model import BrainToTextMamba

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(4, 320, 512, device=device)
lengths = torch.tensor([320, 300, 280, 260], device=device)

model = BrainToTextMamba(
    input_dim=512,
    d_model=256,
    num_layers=6,
    vocab_size=50,
).to(device)

logits, out_lengths = model(x, lengths)

print("logits:", logits.shape)
print("lengths:", out_lengths)
