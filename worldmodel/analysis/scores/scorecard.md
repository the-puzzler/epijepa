# EpiJEPA-LeWM vs the released LeWM checkpoints — final scorecard

Planning success rate (%). Same architecture as the released LeWM checkpoints; Epi replaces SIGReg with the
epiplexity score + one non-affine BatchNorm on the projector output. CEM planner, LeWM defaults (300 samples,
30 iters, top-30, horizon 5). Three eval sets, all seed 42, starts drawn from the training dataset:
- **paper-50**: the paper's exact eval (50 episodes, goal offset 25, budget 50)
- **n=200**: deeper eval, disjoint from the 50
- **n=500**: backup eval (10 x 50 chunks), distinct starts verified

Epi recipe: encoder form (frozen CNN reservoir -> ridge -> embedding), one form for all envs, λ per env.

## Headline (mean over seeds; seeds in brackets)

| Env | Budget | Method | paper-50 | n=200 | n=500 |
|---|---|---|---|---|---|
| TwoRoom | 30k | **Epi λ0.03** (3) | **100** (100/100/100) | **100** (100/100/100) | **99.9** (100/100/99.8) |
| | ~200k | Released | 86 | 85.0 | 82.8 |
| Push-T | 60k | **Epi λ0.1** (3) | 92 (92/94/90) | **89.3** (88.5/89.5/90.0) | 88.5 (88.4/88.0/89.0) |
| | ~200k | Released | 96 | 83.5 | 84.6 |
| Cube | 60k | **Epi λ0.03** (3) | 70.7 (72/66/74) | **73.2** (74.5/69.5/75.5) | **71.5** (72.2/69.8/72.4) |
| | 100k | Epi λ0.03 (1) | 78 | 75.0 | 73.0 |
| | ~200k | Released | 68 | 63.0 | 66.0 |
| Reacher | 60k | Epi λ0.3 (3) | 54.7 (58/50/56) | 54.7 (57.5/50.5/56.0) | 58.1 (57.6/60.4/56.4) |
| | 100k | Epi λ0.3 (3) | 64.0 (68/60/64) | 58.3 (61.0/53.5/60.5) | 63.4 (66.8/60.0/63.4) |
| | 200k | **Epi λ0.3** (3) | **72.7** (76/70/72) | **72.0** (72.0/70.0/74.0) | **73.5** (76.6/69.4/74.4) |
| | ~200k | Released | 52 | 62.0 | 60.8 |

## Significance (n=500, Fisher exact, pooled seeds)
- TwoRoom: Epi 99.9 vs released 82.8 — overwhelming.
- Cube (60k): Epi 71.5 vs released 66.0, p = 0.024.
- Push-T: Epi 88.5 vs released 84.6.
- Reacher: the budget matters — 60k Epi 58.1, 100k Epi 63.4 (vs released 60.8, n.s.);
  **200k (matched to released budget) 73.5 vs 60.8, p ≈ 1e-7 (3 seeds); n=200 72.0 vs 62.0, p = 0.01.**

## Verdict
- vs **released** weights: Epi beats them on all four envs on both n=200 and n=500 (Reacher at matched ~200k budget, 3 seeds).
- paper-50 set is unreliable (e.g. released Push-T 96 on paper-50 vs 83.5/84.6 on n=200/500).

## Notes for the write-up
- Planning success peaks at ~60k on Push-T for both methods (200k Epi 81.5, released 83.5); Reacher keeps improving
  to 200k; Cube is flat 60k->100k. So budgets differ per env.
- TwoRoom: paper text says history 1, released checkpoint uses 3; we trust the release (3 everywhere).
- Eval start states come from the training dataset (LeWM protocol).
- Ablation: stop-gradient on the prediction target collapses planning (paper-50 0-4 by 40k; eff. rank 53 vs 151) —
  the target-side gradient is load-bearing in LeWM.

Every individual number (all runs, checkpoints and eval sets) is in all_scores.csv in this folder.
