Push all changes to GitHub safely and report which Colab notebooks need re-running.

## Steps

1. Run `git status` and `git diff --stat` to see all changes
2. Security check — warn if any of these are staged:
   - `.env`, `credentials.json`, API keys, tokens
   - Files > 10MB (data, checkpoints, model artifacts)
   - `*.pkl`, `*.pt`, `*.pth` files
3. Stage appropriate files (skip anything in `.gitignore`)
4. Generate a clear commit message summarizing the changes
5. Commit and push to `origin main`
6. Based on which files changed, tell the user which notebooks to re-run using this map:

| Changed file pattern | Re-run from |
|---|---|
| `scalp2/data/*` | NB 01 → 02 → 03 → 04 → 05 → 06 |
| `scalp2/features/*` | NB 02 → 03 → 04 → 05 → 06 |
| `scalp2/labeling/*` | NB 03 → 04 → 05 → 06 |
| `scalp2/models/*`, `scalp2/training/trainer.py` | NB 04 → 05 → 06 |
| `scalp2/training/stage2_trainer.py`, `scalp2/regime/*` | NB 05 → 06 |
| `config.yaml` (labeling/model/training sections) | NB 03 → 04 → 05 → 06 |
| `config.yaml` (execution section only) | NB 06 only |
| `scalp2/execution/*` | NB 06 only |
| `notebooks/*.ipynb` only | Just re-run the changed notebook |

7. If no files changed, say "Nothing to push."
