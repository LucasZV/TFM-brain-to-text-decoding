from omegaconf import OmegaConf
from convmamba_text_trainer import BrainToTextDecoderTrainer

args = OmegaConf.load("convmamba_text_args.yaml")
trainer = BrainToTextDecoderTrainer(args)
metrics = trainer.train()