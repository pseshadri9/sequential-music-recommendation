# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code accompanying the RecSys 2023 MuRS talk *"Leveraging Negative Signals with Self-Attention for Sequential Music Recommendation"* (see `README.md` for the paper link). It trains self-attention transformers for next-track recommendation that explicitly model **skip (negative) signals** via a contrastive loss. This is experimental research code, not a packaged library.

## Setup & run

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).

```bash
bash setup.sh                              # installs uv if needed, then `uv sync` creates .venv and installs deps
uv run python main.py config/config_lfm.yml   # train+eval on a chosen config (defaults to config/config_lfm.yml if no arg)
# or: source .venv/bin/activate && python main.py config/config_lfm.yml
```

- `pyproject.toml` declares deps unpinned (`requires-python = ">=3.9"`); `uv.lock` is the resolved, committed lockfile — run `uv lock` after editing deps. `torch`/`torchvision`/`torchaudio` install from the default PyPI index (CUDA build on Linux).
- `[tool.uv] package = false` — this is an application/script repo, not an installable package, so uv doesn't try to build it.
- There is **no test suite, linter, or build step** — `train_evaluate/{train,eval}.py` and `utility/run.py` are empty stubs. The only entry point is `main.py`.
- `run.sh` is a scratch file holding example invocations and a `tensorboard --logdir ...` command (paths are stale). View logs with `tensorboard --logdir logger_runs/<save_dir>`.

### Gotchas before running
- **`main.py` calls `input()` to name the run** unless `dev: True` in the config — a non-interactive run will hang waiting on stdin. Set `dev: True` (uses a 1000-session subset, names the run "dev") for quick local iteration.
- **Config `data_path` / `ckpt_path` values are absolute paths on the original author's machine** (`/home/pavans/...`, `/media/pavans/...`) — edit them before running.
- Configs assume `accelerator: "gpu"`, `devices: [0]`.

## How a run is wired together

`main.py` is the orchestrator. It loads a YAML config, builds a Lightning `DataModule` + `VanillaTransformer`, trains, tests, and writes a JSON manifest. Everything is driven by the config dict — there are no CLI flags beyond the config path.

**Config sections** (`config/*.yml`): `model_params` (passed as `**kwargs` to `VanillaTransformer`), `data_params`, `exp_params.manual_seed`, `trainer_params` (passed to Lightning `Trainer`), `logging_params`, plus top-level `dev` and `dual_train` booleans.

### Data layer (`data_process/process_dataset.py`)
- **DataModule is selected by file extension**: `data_path` ending in `.csv` → `LfMDataModule` (single CSV, Last.fm); otherwise → `SpotifyDataModule` (a *directory* of CSVs, Spotify Sequential Skip Prediction Challenge format). `LfMDataModule` subclasses `SpotifyDataModule` and overrides `load_csv`/`load_data`.
- Sessions are filtered (min length 5, must contain ≥1 skip), tokenized to integer track IDs, zero-padded to `max_seq_len` (20). Skips are binarized.
- `split_data` produces the train/val/test split by **holding out the last items of each session**: test target = last item, val target = 2nd-to-last, last train target = 3rd-to-last. This is why the TensorDatasets have different arities (train: 3 tensors, val: 4, test: 5) and why the model has separate `get_val_batch`/`get_test_batch` helpers that splice held-out targets back into the session.
- **Reserved token indices** (`data_process/constants.py`): `PADDING_ITEM=0`, `CLS_ITEM=1`, `MASKING_ITEM=2`; real track IDs start at index 3 (`NUM_RESERVED_TOKENS`). Skip sequences use their own pad value `SKIP_PAD=2`.
- Skip semantics are dataset-dependent and the inline comments contradict each other — verify against the actual `load_csv`/`skip_preprocess` for the dataset you're touching rather than trusting a single comment.

### Model (`models/models.py` → `VanillaTransformer`)
A single PyTorch Lightning module is the project's core; `main.py` imports it as `from models import VanillaTransformer`.
- **`bidirectional` flag switches the training regime**: `False` = SASRec-style causal (uses a square subsequent mask); `True` = BERT4Rec-style masked-item modeling (`_mask_tracks` randomly masks ~15% of items plus the last position; `max_seq_len` is bumped by 1).
- **`return_skip` flag enables the negative-signal contrastive loss**: when on, an `InfoNCE` loss (`utility/loss_metrics.py`) is computed over skip-derived positive/negative pairs (`get_negative_samples` + `_closest_pos_sample`) and added to the next-item loss, scaled by `wt`. This contrastive use of skips is the paper's central contribution.
- **Sampled softmax** (`sampled_softmax`) is used instead of a full-vocab softmax for tractable training over large track vocabularies; the decoder ties weights to the token embedding (`torch.matmul(x_, self.vocab.weight.T)`).
- **Metrics**: top-K hit rate (`test_top_k`, K from `model_params.k`), `RetrievalMAP@10`, and a skip-specific `RetrievalMRR@10` (computed only over skipped targets). AUROC code exists but is commented out.
- `dual_train: True` triggers a two-phase schedule in `main.py:train()` — first train embeddings with `return_skip=False`, reload best checkpoint, freeze the vocab embedding, then fine-tune with the skip loss on.

### Outputs (`notification/manifest_handler.py`)
After test, `manifestHandler` writes a JSON manifest to `logger_runs/manifest/<logging_params.name>/<run_name>-<n>.json` capturing the config, eval metrics, and best-checkpoint path. TensorBoard event files + checkpoints go under `logging_params.save_dir`. `notification/email.py` (`send_email`) can email results but depends on `notification/email_args.py`, which is gitignored.

## Things that look like code paths but aren't
- `models/baseline_models.py` (Caser, etc.) and `models/baseline_model.py` have **broken imports** (`from constants import`, `from attention import` — no such top-level modules) and are not wired into `main.py`. Treat as WIP/standalone references.
- `models/bidirectional.py` is an older standalone copy of `VanillaTransformer`; the live model is `models/models.py`. Both define a class named `VanillaTransformer` — only the one in `models.py` is exported via `models/__init__.py`.
- Large data artifacts (`*.npy`, `*.zip` at repo root; everything under `datasets/` and `logger_runs/` except `.gitkeep`) are **gitignored** — don't commit them. The `np.save(...)` block in `main.py` that produces the test-sample `.npy` files is commented out.
