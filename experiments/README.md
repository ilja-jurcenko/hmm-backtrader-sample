# Experiment Framework

Structured benchmark for measuring the impact of HMM regime filters on classical trading strategies across multiple configurations and asset universes.

Each experiment runs all 13 strategies in parallel via `walkforward-compare.py`, one sbatch job per *(group × ticker set × timeframe)*.

---

## Directory Structure

```
experiments/
  configs/
    common.yaml         # shared defaults (dates, n_trials, cpus, strategies, timeframes)
    ticker_sets.yaml    # named ticker sets (spy_qqq, sp500_10)
    groups.yaml         # experiment groups with HMM args, phase labels
  gen_jobs.py           # generate sbatch scripts + submit_phase_N.sh from configs
  aggregate.py          # read all *_results.csv → master comparison table
  submit_phase_N.sh     # auto-generated; one per phase (run after gen_jobs.py)
  jobs/
    {group_id}/{ticker_set}/{timeframe}.sh   # one sbatch script per job

results/                # written by jobs; NOT checked in
  {group_id}/{ticker_set}/{timeframe}/
    {strategy}.txt             # walkforward-compare stdout
    {strategy}_windows/        # per-window detail logs
    {strategy}_results.csv     # structured metrics per window

logs/                   # Slurm stdout/stderr (%j = job ID)
  {group__ticker_set__timeframe}_%j.out
```

---

## One-Time Setup

Regenerate all job scripts and submit scripts after editing any config:

```bash
python experiments/gen_jobs.py
```

Use `--dry-run` to preview without writing:

```bash
python experiments/gen_jobs.py --dry-run
```

---

## Launch Plan

Phases run sequentially. Aggregate results between phases to identify winning
configurations before generating the next phase's jobs.

---

### Phase 1 — Regime Mode Sweep (8 jobs)

Tests all four HMM regime modes against a no-HMM baseline on SPY + QQQ across
two timeframes.

| Group | Description |
|---|---|
| `01_strict` | Block all trades in unfavourable states |
| `02_size` | Reduce position size in unfavourable states |
| `03_score` | Per-state position sizes, score-weighted |
| `04_linear` | Linearly scale position by state score |

```bash
bash experiments/submit_phase_1.sh
python experiments/aggregate.py --phase 1
```

→ Identify the best-performing regime mode. Update `--regime-mode` in groups
`05_k2` through `14_best_multiasset` in `configs/groups.yaml`.

---

### Phase 2 — K (Hidden States) Sweep (8 jobs)

Tests K = 2, 3, 4, 6 hidden states using the best regime mode from Phase 1.

```bash
python experiments/gen_jobs.py
bash experiments/submit_phase_2.sh
python experiments/aggregate.py --phase 2
```

→ Pick the best K. Update `--hmm-components` in groups `09_pca_none` through
`14_best_multiasset`.

---

### Phase 3 — PCA Sweep (6 jobs)

Tests no PCA vs. PCA=3 vs. PCA=4 components, using the best mode + K.

```bash
python experiments/gen_jobs.py
bash experiments/submit_phase_3.sh
python experiments/aggregate.py --phase 3
```

→ Pick the best PCA setting. Update groups `12_feat_compact`, `13_feat_full`,
and `14_best_multiasset`.

---

### Phase 4 — Feature Set Sweep (4 jobs)

Compact features (`Returns Range vol log_ret`) vs. full 13-feature set.

```bash
python experiments/gen_jobs.py
bash experiments/submit_phase_4.sh
python experiments/aggregate.py --phase 4
```

→ Pick the winning feature set. Copy the complete winning `hmm_args` into group
`14_best_multiasset`.

---

### Phase 5 — Multi-Asset Validation (4 jobs)

Best config from Phases 1–4 applied to both ticker sets:
- `spy_qqq`: SPY, QQQ
- `sp500_10`: AAPL, MSFT, NVDA, AMZN, JPM, XOM, UNH, WMT, BAC, KO

```bash
python experiments/gen_jobs.py
bash experiments/submit_phase_5.sh
python experiments/aggregate.py --phase 5
```

---

### Final Results

```bash
python experiments/aggregate.py
# → results/master_results.csv
```

**Total: 30 jobs across 5 phases, 16 CPUs per job.**

---

## Updating Configs Between Phases

After aggregating Phase N results, edit `configs/groups.yaml` to fill in the
best values for the next phase's groups (marked with `# TODO` comments), then
regenerate:

```bash
python experiments/gen_jobs.py
```

---

## Strategies Tested

All 13 strategies run in every job:

`sma` · `dema` · `rsi` · `macd` · `adx_dm` · `channel_breakout` · `donchian` ·
`ichimoku` · `parabolic_sar` · `tsmom` · `turtle` · `vol_adj` · `hmm_mr`

---

## HMM Variants Explored

| Dimension | Values |
|---|---|
| Regime mode | `strict`, `size`, `score`, `linear` |
| K (hidden states) | 2, 3, 4, 6 |
| PCA components | none, 3, 4 |
| Feature set | compact (4 features), full (13 features) |
| Baseline | no `--hmm` flag (HMM disabled, `use_hmm=False`) |
