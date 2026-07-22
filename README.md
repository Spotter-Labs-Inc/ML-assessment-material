# Freight Rate Prediction Challenge

Predict `posted_rate` for every load in `test.csv`.

**Files** (`data/`): `train.csv` (labeled), `test.csv` (predict these), `sample_submission.csv` (output format).

**Submit:**
1. Your code via GitHub
2. `predictions.csv` with columns `load_id,predicted_rate`.
3. Performance metrics: MAE, MAPE, RMSE, R².
4. A loom video explaining your solution.

**Self-score** against your own holdout (needs `load_id,posted_rate`):

```bash
pip install -r requirements.txt
python score.py --predictions predictions.csv --labels your_holdout.csv
```
