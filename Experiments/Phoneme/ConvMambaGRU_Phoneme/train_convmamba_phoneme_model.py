from omegaconf import OmegaConf
from convmamba_phoneme_trainer import BrainToPhonemeDecoderTrainer

args = OmegaConf.load("convmamba_phoneme_args.yaml")
trainer = BrainToPhonemeDecoderTrainer(args)
metrics = trainer.train()