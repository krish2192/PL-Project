# EPL Match Outcome Predictor

Predict English Premier League results (Home / Draw / Away) with an end-to-end pipeline: ingest messy season CSVs, engineer leakage-safe features, train models, serve a FastAPI endpoint, and demo everything in a small web client.

## What this project does
- Cleans football-data.co.uk season files into a single tidy table with stable match IDs and harmonized team names.
- Builds pre-match features: rolling form (goals/shots/points), rest days, Elo with home advantage, and implied probabilities from bookmaker odds.
- Trains and evaluates models with time-aware cross-validation: baseline multinomial logistic regression plus an optional LightGBM booster.
- Serves predictions via FastAPI (`/health`, `/teams`, `/limits`, `/predict`) and a simple browser UI that consumes the API.

## Run it in 6 steps
```bash
# 0) Python env
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pandas numpy pyarrow scikit-learn joblib fastapi uvicorn[standard] lightgbm

# 1) Ingest raw season CSVs (place E0.csv files under data/raw/epl/**/)
python data_ingest.py --input-root data/raw/epl --pattern "E0.csv" \
  --out-csv data/processed/epl_matches.csv --out-parquet data/processed/epl_matches.parquet

# 2) Build features (rolling form, rest, Elo, odds)
python build_features.py --input data/processed/epl_matches.parquet --out data/processed/epl_features.parquet

# 3a) Train baseline logistic regression
python train_baseline.py --features data/processed/epl_features.parquet --model-out models/baseline_logreg.pkl

# 3b) (Optional) Train LightGBM
python train_lgbm.py --features data/processed/epl_features.parquet --model-out models/lgbm.pkl

# 4) Serve the API (reload for dev)
uvicorn api_server:app --reload --port 8000

# 5) Smoke-test the endpoints
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/teams"
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
  -d '{"home_team":"Arsenal","away_team":"Chelsea","date":"2025-08-12"}'

# 6) Open the web client
# Open web/index.html in your browser and point the API Base URL to http://127.0.0.1:8000
```

## How it works (plain English)
1) Ingestion (`data_ingest.py`): read season CSVs, fix column names, standardize team aliases, parse dates, and create `match_id`. Output: `data/processed/epl_matches.{csv,parquet}`.
2) Feature engineering (`build_features.py`): for each match, compute prior rolling stats per team, rest days, Elo ratings before kickoff, and implied odds. Output: `data/processed/epl_features.{csv,parquet}` with target `y` (0=Home, 1=Draw, 2=Away).
3) Modeling (`train_baseline.py`, `train_lgbm.py`): time-series cross-validation to avoid leakage; save fitted pipelines to `models/`.
4) Serving (`api_server.py`): loads matches and the latest model, rebuilds features on-the-fly for a requested date/teams, then returns calibrated probabilities.
5) UI (`web/index.html`, `web/main.js`): calls the API, lets you pick teams/date, and shows probability bars.

## Files at a glance
- `data_ingest.py` — normalize raw football-data.co.uk CSVs, harmonize teams, add season, build match IDs.
- `build_features.py` — leakage-safe rolling form, rest-diff, Elo, implied odds; writes Parquet and CSV twins.
- `train_baseline.py` — multinomial logistic regression with TimeSeriesSplit; saves `models/baseline_logreg.pkl`.
- `train_lgbm.py` — LightGBM multiclass with time-aware CV; saves `models/lgbm.pkl`.
- `api_server.py` — FastAPI app exposing `/health`, `/teams`, `/limits`, `/predict`.
- `web/index.html`, `web/main.js` — static client hitting the API.
- `data/raw/epl/` — put source CSVs here (ignored).
- `data/processed/`, `models/` — generated outputs and trained pipelines.

## Feature engineering cheat sheet
- Rolling windows (3, 5, 10): goals for/against, shots, shots on target, points; all shifted to avoid using the current match.
- Rest & scheduling: `home_days_since_last`, `away_days_since_last`, and `rest_diff`.
- Elo: simple Elo with configurable `elo_k` and `elo_home_adv`; ratings stored as pre-match values.
- Betting odds: Bet365 odds converted to implied probabilities when present.
- Target: `y` in {0: Home, 1: Draw, 2: Away}.

## Modeling notes
- Chronological validation via `TimeSeriesSplit`; metrics: accuracy, log loss, macro F1 printed per fold and averaged.
- Pipelines include imputers + one-hot encoders to handle missing odds, early-season gaps, and unseen teams/seasons.
- If you add a calibrated bundle at `models/lgbm_calibrated.pkl`, the API will prefer it.

## API surface
- `GET /health` — service status and loaded data counts.
- `GET /teams` — list of known team names.
- `GET /limits` — min/max allowable dates (based on available data).
- `POST /predict` — body: `{home_team, away_team, date}`; returns probabilities for Home/Draw/Away and `top_class`.

## Resume-ready talking points
- Built an end-to-end sports analytics stack: raw ingest → feature engineering (rolling form, rest, Elo) → time-aware CV → FastAPI → web demo.
- Focused on leakage prevention and chronological evaluation for realistic metrics.
- Delivered calibrated probability outputs with a lightweight client for quick stakeholder demos.

## Nice-to-haves if you keep going
- Track experiments in MLflow or W&B.
- Add a Dockerfile/docker-compose to ship API + static web together.
- Write unit tests for feature builders and a `/predict` happy-path.
- Schedule weekly data refresh and retrain.
