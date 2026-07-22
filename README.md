# Freight Rate Prediction — Take-Home

Predict `posted_rate` (total dollars) for every load in `test.csv`. Timebox: **3–5 hours**.

**Files** (`data/`): `train.csv` (labeled), `test.csv` (predict these), `sample_submission.csv` (output format).
Note: `dph` is blank in `test.csv` — it isn't known at prediction time.

**Submit:**
1. Your code.
2. `predictions.csv` with columns `load_id,predicted_rate`.
3. Your held-out MAE, MAPE, RMSE, R².
4. A ≤1-page writeup: validation approach, what drives rates, weaknesses / next steps.

**Self-score** against your own holdout (needs `load_id,posted_rate`):

```bash
pip install -r requirements.txt
python score.py --predictions predictions.csv --labels your_holdout.csv
```
