"""One-off helper: load the Spotify data once, report vocab/session counts,
then search for the largest batch size that fits on this GPU using the heavier
return_skip=True training path. Not part of the normal pipeline."""
import time, datetime
import torch
from torch.utils.data import DataLoader
from data_process import SpotifyDataModule
from models import VanillaTransformer

torch.set_float32_matmul_precision('high')

DATA = "/workspace/data"

t0 = time.time()
print(f"[{datetime.datetime.now():%H:%M:%S}] loading data from {DATA} ...", flush=True)
data = SpotifyDataModule(DATA, batch_size=64, dev=False)
print(f"load took {time.time()-t0:.0f}s", flush=True)
print("vocab_size:", len(data.vocab), flush=True)
print("train sessions:", len(data.train_data), flush=True)
print("val sessions:", len(data.val_data), "test sessions:", len(data.test_data), flush=True)

model_kwargs = dict(max_seq_len=20, h_dim=128, nhead=8, token_dim=128,
                    nEncoders=4, dropout=0.2, k=[1, 5, 10, 20], lr=0.005,
                    bidirectional=False, wt=0.5)

def trial(bs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = VanillaTransformer(**model_kwargs, vocab_size=len(data.vocab), return_skip=True).cuda()
    dl = DataLoader(data.train_data, bs, shuffle=True, num_workers=2)
    batch = next(iter(dl))
    batch = [x.cuda() for x in batch]
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    opt.zero_grad()
    loss = model.training_step(batch, 0)
    loss.backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, dl, batch, loss, opt
    torch.cuda.empty_cache()
    return peak

import sys
candidates = [int(x) for x in sys.argv[1:]] or [224]
best = None
for bs in candidates:
    try:
        peak = trial(bs)
        print(f"batch {bs:5d}: peak {peak:6.2f} GB  OK", flush=True)
        best = bs
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"batch {bs:5d}: OOM", flush=True)
            torch.cuda.empty_cache()
            break
        raise
print("LARGEST_OK_BATCH:", best, flush=True)
