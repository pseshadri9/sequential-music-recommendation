# Research Log

## Bootstrap / infrastructure (consolidated from the manual campaign)
- Env: converted conda→uv; torch 2.11+cu128 for RTX 5090 (Blackwell sm_120); wandb logger added (config-selectable vs TensorBoard).
- Data: validated `/workspace/data` = Spotify Skip-Prediction logs; loader samples ~561k sessions / 628k vocab; batch-size tuned to 288 (cold-start max, expandable_segments).

## Methodology fixes (pre-registered as code changes)
- LR warmup (`warmup_steps`): depth-4 diverges at lr 0.005 without it.
- MAP fix: `RetrievalMAP` needs non-negative scores → `output.exp()`; was 0.0.
- MRR/MAP index fix: per-epoch global offset (per-batch `arange` collided → MAP 0 / MRR ~1.0).
- Deterministic vocab: `enumerate(sorted(vocab))` (PYTHONHASHSEED non-determinism broke re-eval).
- Dual-best eval: test best-by-val_loss AND best-by-Val-top1, in-process; capture both ckpt paths before testing (test() restores callback state).
- Per-run checkpoint subdirs (provenance).

## Inner loop — experiments
- Capacity: 2L vs 4L; naive 4L "worse" was an optimization failure → fixed via warmup.
- Loss combination ablations: (a) InfoNCE projection head — hurt; (b) Kendall uncertainty weighting — ~neutral.
- **wt sweep [0.1–1.0]** (dual-best): inverted-U, peak **wt≈0.75 (top-1 0.4041)**.
- Annealing (0→wt over 2000 steps): hurt (~0.36).
- Gradient surgery: PCGrad (0.396), CAGrad (running) — neither beats tuned wt.

## Outcomes so far
- H2 (wt sweet-spot) SUPPORTED; H3 (conflict-mitigation) REFUTED; H4 (anneal) REFUTED; H1 (skip helps) weakly supported pending fair best-ckpt baseline + multi-seed.

## Pending
- CAGrad eval; re-run plain-skip wt=0.5; retroactive best-ckpt no-skip baseline; outer-loop synthesis; progress report; conclude → paper.

## Autoresearch formalization
- Set up /loop (cron a9678b9b, 20m) and workspace state files. Continuing the campaign under the two-loop framework.
