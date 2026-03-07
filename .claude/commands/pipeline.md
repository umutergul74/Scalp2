Determine which Colab notebooks need to be re-run based on recent code changes.

## Steps

1. Run `git diff --name-only HEAD~1` (or compare with the user's last known good run)
2. Map each changed file to the notebook dependency chain:

```
scalp2/data/*                    → 01 → 02 → 03 → 04 → 05 → 06
scalp2/features/*                →      02 → 03 → 04 → 05 → 06
scalp2/labeling/*                →           03 → 04 → 05 → 06
scalp2/models/hybrid.py          →                04 → 05 → 06
scalp2/models/meta_learner.py    →                     05 → 06
scalp2/training/trainer.py       →                04 → 05 → 06
scalp2/training/stage2_trainer.py →                    05 → 06
scalp2/training/walk_forward.py  →                04 → 05 → 06
scalp2/regime/hmm.py             →                     05 → 06
scalp2/execution/*               →                          06
config.yaml (labeling section)   →           03 → 04 → 05 → 06
config.yaml (model/training)     →                04 → 05 → 06
config.yaml (execution only)     →                          06
config.yaml (regime)             →                04 → 05 → 06
notebooks/0X_*.ipynb             →  just that notebook
```

3. Find the EARLIEST notebook that needs re-running
4. Report the sequence: "Run NB XX → then XX → then XX"
5. Estimate Colab GPU hours if relevant:
   - NB 04 (Stage 1): ~2-4 hours on T4
   - NB 05 (Stage 2): ~30-60 minutes on T4
   - NB 06 (Backtest): ~5 minutes
   - Others: < 5 minutes each

6. If only `config.yaml` execution section or `scalp2/execution/*` changed, emphasize that ONLY notebook 06 needs re-running (no retraining needed).
