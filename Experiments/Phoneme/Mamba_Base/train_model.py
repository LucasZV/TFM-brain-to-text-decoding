from omegaconf import OmegaConf
from mamba_trainer import BrainToTextDecoder_Trainer

args = OmegaConf.load('mamba_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()
