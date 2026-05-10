from omegaconf import OmegaConf
from gru_text_trainer import BrainToTextDecoderTrainer

args = OmegaConf.load("gru_text_args.yaml")
trainer = BrainToTextDecoderTrainer(args)
metrics = trainer.train()