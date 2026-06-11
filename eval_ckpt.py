"""Test-only re-evaluation of a saved checkpoint with the current (fixed) code.
Usage: python eval_ckpt.py <config.yml> <checkpoint.ckpt>
Reloads the Spotify data, restores the model, and runs trainer.test — used to
recompute metrics (e.g. the fixed MAP/MRR) without retraining."""
import sys, yaml, torch
from pytorch_lightning import Trainer, seed_everything
from data_process import SpotifyDataModule, LfMDataModule
from models import VanillaTransformer

cfg_path, ckpt_path = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    config = yaml.safe_load(f)

seed_everything(config['exp_params']['manual_seed'])
torch.set_float32_matmul_precision('high')

dp = config['data_params']['data_path']
if dp.endswith('.csv'):
    data = LfMDataModule(dp, config['data_params']['batch_size'], dev=config['dev'])
else:
    data = SpotifyDataModule(dp, config['data_params']['batch_size'], dev=config['dev'])

model = VanillaTransformer.load_from_checkpoint(ckpt_path)
model.eval()

trainer = Trainer(logger=False, accelerator='gpu', devices=[0])
print(trainer.test(model, dataloaders=data.test_dataloader()))
