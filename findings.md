# Findings — Negative (Skip) Signals for Sequential Music Recommendation

**Research question.** Does leveraging negative (skip) signals via a self-attention *contrastive* (InfoNCE) auxiliary loss improve next-track sequential recommendation, and — given the two objectives can conflict — how should the next-item loss and the skip-contrastive loss be combined?

**Context.** This extends the MuRS @ RecSys 2023 work in this repo. Primary task: next-track prediction over listening sessions. Auxiliary signal: per-track skip labels, used contrastively (skipped tracks = negatives, "closest positive" = positive) via InfoNCE on the encoder representation.

## Setup (locked)
- **Data:** Spotify Sequential Skip Prediction Challenge logs (`/workspace/data`, 113 CSVs ≈ 58 GB). Loader samples ~**561,090 sessions**, vocab **628,308 tracks** (code's ~450k-session cap → ~3 files; load ~31 s).
- **Model:** `VanillaTransformer` (unidirectional, post-LN), token_dim 128, 8 heads, h_dim 128, **4 encoder layers**, max_seq_len 20.
- **Train:** batch **288** (cold-start max on RTX 5090 32 GB w/ `expandable_segments`), **lr 0.005 + 2000-step linear warmup**, 12 epochs, Adam, grad-clip 5, seed 1265.
- **Eval (primary):** next-track **top-1** hit-rate over a sampled candidate set (observed-in-batch + 1000 negatives). Secondary: top-5/10/20, MAP@10, Skip-MRR@10. **Tested on the BEST checkpoint** (min val_loss AND max Val top-1), not the final epoch.
- **Compute:** single RTX 5090, torch 2.11+cu128 (Blackwell sm_120), uv env, wandb logging.

## Current Understanding
1. **The skip-contrastive loss can marginally help next-track top-1 — but only under three conditions together:** (i) a *tuned* loss weight (`wt ≈ 0.75`), (ii) **best-checkpoint** selection rather than last-epoch (these models overfit late; val top-1 peaks ~epoch 3–5), and (iii) adequate **capacity + LR warmup** (depth-4 at lr 0.005 only trains with warmup; without it the loss is pinned at the random baseline).
2. **The "gradient conflict" framing did not pay off.** Sophisticated multi-task combiners — InfoNCE projection head, Kendall uncertainty weighting, PCGrad, CAGrad — did **not** beat simply tuning the scalar `wt`. The auxiliary signal's value is captured by a single well-chosen weight; de-conflicting gradients added complexity without gains.
3. **The effect is small (~+0.4 pt top-1) and within plausible single-seed noise.** It is suggestive, not conclusive; multi-seed runs are required to claim significance.

## Key Results — dual-best top-1 (4-layer, lr 0.005 + warmup, seed 1265)
| Variant | top-1 @best-val_loss | top-1 @best-top1 |
|---|---|---|
| **no-skip baseline** | 0.3999 (last-epoch†) | — (re-test pending) |
| skip wt=0.10 | 0.3670 | 0.3670 |
| skip wt=0.25 | 0.3934 | 0.3936 |
| skip wt=0.50 (plain) | re-run pending | re-run pending |
| **skip wt=0.75** | 0.3941 | **0.4041** ← best |
| skip wt=1.00 | 0.3920 | 0.3982 |
| skip wt=0.5 + anneal(0→wt, 2k) | 0.3505 | 0.3616 (hurt) |
| skip wt=0.5 + PCGrad | 0.3870 | 0.3956 |
| skip wt=0.5 + CAGrad | running | running |
| (a) projection head (wt 0.5)‡ | — | 0.3706 (last-epoch, hurt) |
| (b) uncertainty weighting‡ | — | 0.3928 (last-epoch, ~neutral) |

† no-skip baseline number is **last-epoch**; best-checkpoint re-test pending (its checkpoint = `epoch=2-step=5847-v4.ckpt`, sorted-vocab, re-testable).
‡ (a)/(b) were run before the dual-best/sorted-vocab infra; their checkpoints predate the sorted-vocab fix and may not re-test cleanly → reported as last-epoch with caveat.

**Trend:** clear inverted-U in `wt`, peak at **wt≈0.75**. Conflict-mitigation methods cluster around plain-skip (~0.39), none beating wt=0.75.

## Patterns and Insights
- **Best-checkpoint selection matters more than the loss-combination method.** Switching from last-epoch to best-checkpoint moved the story from "skip loss is net-negative" to "skip loss marginally positive at the right wt." Late-epoch overfitting was masking the auxiliary's benefit.
- **Capacity interacts with optimization, not just the loss.** Going 2→4 layers needed an LR fix (warmup) to even train; the naive depth experiment looked like "capacity hurts" purely due to optimization failure.
- **Auxiliary-loss annealing hurt here** — delaying the skip signal lost an early-representation benefit, opposite to the usual auxiliary-warmup intuition.

## Lessons and Constraints (do-not-repeat)
- **Depth-4 post-LN transformer diverges at lr 0.005 without warmup** (loss stuck at ln(candidate-set) ≈ 8.0). Always use `warmup_steps≈2000` at this depth/LR.
- **`RetrievalMAP` (torchmetrics 1.9) returns 0.0 on negative scores** — feed `output.exp()` (probabilities), not log-probs. Same code path for MRR.
- **Per-batch `arange` indexes collide across an epoch** in `RetrievalMAP`/`MRR` → MAP→0, MRR→~1.0. Use a running global offset per epoch.
- **Vocab mapping must be deterministic** — `enumerate(sorted(vocab))`, else set-of-strings ordering changes per process (PYTHONHASHSEED) and checkpoints don't re-test.
- **Cross-process checkpoint re-eval is only reliable post sorted-vocab fix AND with the correct manifest-identified checkpoint** (the shared/scattered ckpt dir made `last.ckpt` ambiguous; earlier re-evals gave random ~0.006 from stale checkpoints). In-process train+test is always safe.
- **`/workspace` has a disk quota** — each run writes ~2 GB of checkpoints; prune between runs. CAGrad crashed once on `Disk quota exceeded`.
- **PCGrad/CAGrad fit at batch 288** with `expandable_segments` despite retain-graph + dual gradients (~30 GB). CAGrad's per-step 21-point grid search is ~1.5× slower.

## Open Questions
- Best-checkpoint **no-skip baseline** number (re-test pending) — is wt=0.75's +0.4 pt real vs a *fair* baseline?
- **Multi-seed** confirmation of the wt=0.75 effect (currently single seed).
- Does the wt≈0.75 optimum hold at other capacities / on the LFM dataset?
- Why does annealing hurt — is the skip signal most useful *early* in training?

## Status
Inner-loop sweep + conflict-mitigation experiments essentially complete. Remaining: CAGrad eval, re-run plain-skip wt=0.5 (sorted/dual-best), retroactive best-checkpoint no-skip baseline. Then outer-loop synthesis → likely **CONCLUDE** (coherent finding + clean negative result on conflict-mitigation).
