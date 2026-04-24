# EPL Match Outcome Predictor — Technical Project Documentation

> **A production-grade machine learning system for predicting English Premier League match outcomes, featuring real-time bookmaker odds integration, expected value analysis, and a full-stack web interface.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Engineering Pipeline](#data-engineering-pipeline)
4. [Machine Learning Model](#machine-learning-model)
5. [Expected Value (EV) & Kelly Criterion Engine](#expected-value-ev--kelly-criterion-engine)
6. [Backend — FastAPI](#backend--fastapi)
7. [Frontend — Next.js](#frontend--nextjs)
8. [Deployment Architecture](#deployment-architecture)
9. [Key Engineering Decisions](#key-engineering-decisions)
10. [API Reference](#api-reference)
11. [Project Structure](#project-structure)
12. [Setup & Running Locally](#setup--running-locally)
13. [Results & Performance](#results--performance)

---

## Project Overview

The EPL Match Outcome Predictor is a full-stack machine learning application that predicts the outcome (Home Win / Draw / Away Win) of English Premier League fixtures. The system integrates live bookmaker odds from the OddsPapi API, computes the ensemble model's probability estimates for each outcome, and identifies value betting opportunities where the model's probabilities exceed the market's implied probabilities by more than 5%.

### What It Does

- **Predicts match outcomes** using a trained ensemble model (CatBoost + Logistic Regression) with rolling team statistics and Elo ratings as features
- **Fetches live B365 odds** for upcoming fixtures and computes Expected Value (EV) and Kelly Criterion stake sizing for each outcome
- **Exposes a REST API** via FastAPI, with endpoints for historical matches, single-match prediction, and bulk upcoming fixture predictions
- **Visualises predictions** in a Next.js frontend deployed on Vercel, showing probability bars, confidence scores, EV tables, and value bet highlights
- **Stores historical match data** in a Supabase (PostgreSQL) database, enabling rolling feature computation across multiple seasons

### Why This Project

Sports betting markets are largely efficient, but edges exist when a model's probability estimates diverge from the bookmaker's implied probabilities after accounting for the vig (bookmaker's margin). This project quantifies that edge using the Expected Value formula:

```
EV = (model_probability × decimal_odd) - 1
```

A positive EV above a 5% threshold signals a value bet. The Kelly Criterion then determines the theoretically optimal stake fraction to maximise long-run bankroll growth.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│  Supabase (PostgreSQL)    OddsPapi API    Historical CSV files  │
└────────────┬──────────────────┬───────────────────┬────────────┘
             │                  │                   │
             ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND (Python)                      │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Feature Service │  │  Odds Service    │  │ Model Service │  │
│  │  (rolling stats, │  │  (OddsPapi,      │  │ (CatBoost +   │  │
│  │   Elo ratings)   │  │   B365 odds,     │  │  LR ensemble, │  │
│  │                  │  │   EV/Kelly calc) │  │  imputer)     │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬────────┘  │
│           └────────────────────┼───────────────────┘           │
│                                ▼                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              REST API Endpoints                          │  │
│  │  GET  /health                                            │  │
│  │  GET  /api/v1/matches          (historical data)         │  │
│  │  POST /api/v1/predict          (single match)            │  │
│  │  GET  /api/v1/predict/upcoming (bulk + live odds)        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │  HTTP / JSON
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   NEXT.JS FRONTEND (TypeScript)                  │
│                                                                 │
│  /upcoming  → UpcomingCard grid with EV tables & value bets     │
│  /predict   → Manual prediction form with odds input            │
│  /results   → Historical results with accuracy stats            │
└─────────────────────────────────────────────────────────────────┘
```

**Hosting:**
- Backend → Hugging Face Spaces (Docker)
- Frontend → Vercel (Next.js edge deployment)
- Database → Supabase (managed PostgreSQL)

---

## Data Engineering Pipeline

### Data Sources

Historical EPL match data spanning the 2010/11 to 2024/25 seasons was sourced and loaded into Supabase. Each match record contains:

- Full-time result (FTR), half-time result
- Goals scored/conceded (home and away)
- Shots, shots on target, corners, fouls, yellow/red cards
- B365 opening odds (home, draw, away)
- Referee information

### Feature Engineering

The core insight in sports prediction is that raw match data is not predictive — what matters is **recent form and momentum**. All features are computed as rolling windows over recent matches rather than season-wide averages.

**Rolling statistics (last 5 matches):**
- Goals scored per game (home context)
- Goals conceded per game (away context)
- Points accumulated (form trajectory)
- Win rate (historical, season-weighted)
- Days since last match (fatigue/rest proxy)
- Shots on target ratio

**Elo Rating System:**
Each team maintains a dynamic Elo rating updated after every match. The Elo difference between home and away teams (`home_elo - away_elo`) is one of the strongest single predictors in the feature set. The update formula applies a K-factor of 32 with margin-of-victory weighting.

**Odds-derived features (when available):**
When bookmaker odds are provided, three additional features are computed:
- `odds_fair_h` — implied probability of home win (vig-adjusted)
- `odds_fair_d` — implied probability of draw (vig-adjusted)
- `odds_fair_a` — implied probability of away win (vig-adjusted)
- `odds_home_edge` — difference between home and away implied probabilities

These features encode market consensus, which is itself highly informative and acts as a powerful regulariser on the model's raw output.

**Feature imputation:**
A `SimpleImputer` (median strategy) handles missing values for teams with insufficient historical data (e.g., newly promoted clubs). The fitted imputer is serialised alongside the model to ensure identical preprocessing at inference time.

### Data Storage

All historical matches are stored in Supabase with row-level timestamps for reproducibility. The feature engineering pipeline can be triggered on demand to regenerate the `features.parquet` file, which is used as a fast-access cache for inference. If the parquet file is absent, the pipeline falls back to querying Supabase directly.

---

## Machine Learning Model

### Model Architecture

The system uses a **soft-voting ensemble** of two base models:

| Model | Role | Strengths |
|---|---|---|
| **CatBoost Classifier** | Primary learner | Handles categorical features, robust to missing data, strong on tabular data |
| **Logistic Regression** | Secondary learner | Provides well-calibrated probability estimates, acts as regulariser |

The ensemble combines probabilities via weighted averaging, with CatBoost weighted at 0.65 and Logistic Regression at 0.35. This blending reduces CatBoost's tendency to produce overconfident probabilities while retaining its superior discriminative power.

### Target Variable

Three-class classification:
- `H` — Home Win
- `D` — Draw
- `A` — Away Win

### Training

The model is trained on all available historical data up to the current season. A time-based train/validation split is used (no random shuffling) to prevent data leakage — future matches must never inform predictions for past matches.

**Hyperparameter tuning** was performed using Optuna with 50 trials, optimising for macro F1-score on the validation set. Key tuned parameters include:
- CatBoost: `depth`, `learning_rate`, `l2_leaf_reg`, `iterations`
- Logistic Regression: `C` (regularisation strength), `solver`

### Serialised Artefacts

Three files are saved to `models/saved/` and loaded at FastAPI startup:

```
models/saved/
├── ensemble.pkl       ← trained VotingClassifier
├── imputer.pkl        ← fitted SimpleImputer
└── feature_names.pkl  ← ordered list of feature columns
```

Loading all three at startup (not per-request) ensures prediction latency is bounded by feature computation, not model loading.

---

## Expected Value (EV) & Kelly Criterion Engine

### Expected Value

For each outcome, EV quantifies whether a bet at the bookmaker's offered odds is profitable in expectation given the model's probability:

```
EV(outcome) = (model_probability × decimal_odd) - 1
```

- `EV > 0.05` (5%) → **value bet** — the model believes this outcome is underpriced
- `EV > 0` but `< 0.05` → slight edge, not flagged as value
- `EV < 0` → no edge, bookmaker has the advantage

### Bookmaker Vig

The bookmaker's margin (vig) is the amount extracted from each betting market:

```
vig = (1/odd_h + 1/odd_d + 1/odd_a) - 1
```

B365 typical vig for EPL matches is 4–8%. The vig-adjusted implied probabilities are computed by normalising the raw implied probabilities to sum to 1.0.

### Kelly Criterion

For value bets, the Kelly Criterion determines the theoretically optimal fraction of bankroll to stake:

```
Kelly% = edge / (odd - 1)
       = ((model_prob × odd) - 1) / (odd - 1)
```

All Kelly outputs are capped at 15% of bankroll to account for model uncertainty and prevent ruin from overconfident staking. This is known as **fractional Kelly** and is standard practice in applied betting systems.

### Output Structure

Every prediction response includes a full EV analysis object:

```json
{
  "ev_analysis": {
    "bookmaker_vig": 0.0496,
    "has_value": true,
    "best_bet": {
      "outcome": "D",
      "model_prob": 0.3382,
      "decimal_odd": 3.60,
      "ev": 0.2175,
      "kelly_pct": 8.37,
      "is_value": true
    },
    "value_bets": [...],
    "all_outcomes": [...]
  }
}
```

---

## Backend — FastAPI

### Structure

```
app/
├── api/v1/endpoints/
│   ├── predict.py       ← POST /predict, GET /predict/upcoming
│   └── matches.py       ← GET /matches
├── services/
│   ├── feature_service.py  ← rolling stats, Elo, feature matrix
│   ├── odds_service.py     ← OddsPapi, B365 odds, EV/Kelly
│   └── ev_service.py       ← value bet identification
├── schemas/
│   └── prediction.py    ← Pydantic request/response models
├── ml/features/
│   └── feature_columns.py  ← canonical FEATURE_COLS list
└── utils/
    └── team_utils.py    ← team name normalisation map
```

### Key Design Decisions

**Single model load at startup:** The ensemble, imputer, and feature names are loaded once when FastAPI starts using module-level globals. This avoids disk I/O on every request and keeps P99 prediction latency under 300ms.

**Shared `_run_prediction` helper:** Both `POST /predict` and `GET /predict/upcoming` use the same core inference function. This eliminates code duplication and ensures identical behaviour for manual predictions and bulk upcoming predictions.

**Graceful degradation:** If OddsPapi is unreachable, the upcoming endpoint returns whatever fixtures could be fetched. If a specific team cannot be found in the historical data, that fixture is skipped with a warning log rather than failing the entire request.

**Rate limiting compliance:** OddsPapi enforces rate limits. A 1.5-second delay is inserted between per-fixture odds fetches to stay within the allowed call rate.

### CORS Configuration

CORS is configured to allow requests from `http://localhost:3000` (development) and the production Vercel domain. This is required because the Next.js frontend makes browser-side fetch calls directly to the FastAPI backend.

---

## Frontend — Next.js

### Stack

- **Next.js 15** with App Router and TypeScript
- **Tailwind CSS v4** (CSS-first configuration, no `tailwind.config.ts`)
- **shadcn/ui** component library (Card, Badge, Button, Input, Label, Skeleton)
- **Vercel** for deployment

### Pages

| Route | Type | Purpose |
|---|---|---|
| `/upcoming` | Server Component + Suspense | Fetches all upcoming fixtures with predictions and EV |
| `/predict` | Client Component | Manual prediction form with odds input and validation |
| `/results` | Server Component + Suspense | Historical match results with model accuracy stats |

### Component Architecture

```
src/
├── app/
│   ├── upcoming/page.tsx    ← GET /predict/upcoming → UpcomingCard grid
│   ├── predict/page.tsx     ← POST /predict → PredictionResult
│   └── results/page.tsx     ← GET /matches → MatchCard grid + AccuracyStats
├── components/match/
│   ├── MatchCard.tsx        ← shared card for both results and predict output
│   ├── PredictionBadge.tsx  ← H/D/A probability bar
│   ├── OddsRow.tsx          ← B365 odds display
│   └── EVTable.tsx          ← per-outcome EV/Kelly table
├── components/stats/
│   └── AccuracyStats.tsx    ← model accuracy summary
├── lib/
│   ├── api.ts               ← typed API client (all fetch calls)
│   └── errors.ts            ← HTTP status → user-friendly message map
└── types/
    └── index.ts             ← all TypeScript interfaces
```

### Key UI Rules

**Confidence colouring** signals prediction strength at a glance:
- ≥ 50% confidence → green bold
- 40–50% → amber bold
- < 40% → muted grey

**EV colouring** per outcome row:
- `EV > 5%` → green highlight (value bet)
- `EV > 0%` → neutral
- `EV < 0%` → muted red

**Value bets float to the top** of the upcoming fixtures grid, ensuring the most actionable predictions are immediately visible.

**Odds validation** on the predict form enforces an all-or-nothing rule — if any odd is entered, all three must be provided and each must be between 1.01 and 100. This prevents partial EV computations that would produce misleading results.

### Rendering Strategy

Pages that fetch live backend data use `export const dynamic = "force-dynamic"` to opt out of Next.js static pre-rendering at build time. This is necessary because Vercel's build servers cannot reach the FastAPI backend during `next build`. The pages render fresh on every request at runtime instead.

---

## Deployment Architecture

### Backend — Hugging Face Spaces (Docker)

The FastAPI backend is containerised with Docker and deployed to Hugging Face Spaces. The `Dockerfile` installs Python dependencies, copies the trained model artefacts, and starts Uvicorn on port 7860 (the Spaces default).

Hugging Face free tier spaces sleep after inactivity. A `/ping` endpoint is exposed to allow the frontend to wake the space before making data requests.

### Frontend — Vercel

The Next.js frontend is deployed to Vercel via GitHub integration. Every push to `main` triggers an automatic redeploy. The production backend URL is set as `NEXT_PUBLIC_API_URL` in Vercel's environment variable dashboard.

### Environment Variables

| Variable | Location | Value |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Vercel dashboard | `https://your-space.hf.space` |
| Supabase credentials | Backend `.env` / HF Secrets | Supabase project URL + anon key |
| OddsPapi key | Backend `.env` / HF Secrets | API key from OddsPapi dashboard |

---

## Key Engineering Decisions

### Why CatBoost over XGBoost or LightGBM

CatBoost was chosen as the primary ensemble member because it handles categorical features (team names, referees) natively without requiring manual encoding. It also produces strong results on small-to-medium tabular datasets without extensive hyperparameter tuning, and its built-in support for ordered boosting reduces overfitting on time-series data.

### Why Supabase over a flat file store

While a CSV file would suffice for batch predictions, Supabase enables rolling feature computation across arbitrary date windows via SQL queries. It also allows the pipeline to be re-run incrementally as new match results come in, without reprocessing the entire historical dataset.

### Why `force-dynamic` instead of ISR

Incremental Static Regeneration (ISR) would allow pages to be cached and revalidated on a schedule. However, since the backend may be sleeping (Hugging Face free tier), ISR could cache stale data or error states. `force-dynamic` ensures every page load fetches fresh predictions, at the cost of slightly higher time-to-first-byte.

### Why odds features are optional

Making odds optional in `POST /predict` allows the endpoint to be useful even without bookmaker data — for example, when predicting hypothetical fixtures or when OddsPapi is unavailable. The EV analysis section simply returns `null` when odds are absent, and the frontend handles this gracefully by hiding the EV table.

### Why a shared `_run_prediction` helper

The same inference logic powers both the manual `/predict` endpoint and the bulk `/predict/upcoming` endpoint. Extracting it into a shared helper ensures that any improvement to the prediction pipeline — better feature computation, updated model, improved Elo logic — is automatically reflected in both endpoints without code duplication.

---

## API Reference

### `GET /health`

Returns backend status.

**Response:**
```json
{ "status": "ok" }
```

---

### `GET /api/v1/matches`

Returns paginated historical match data.

**Query params:** `limit` (default 50), `offset` (default 0), `season` (optional)

**Response:**
```json
{
  "count": 50,
  "data": [
    {
      "id": 2,
      "date": "2010-08-14",
      "home_team": "Blackburn",
      "away_team": "Everton",
      "ftr": "H",
      "fthg": 1,
      "ftag": 0,
      "b365h": 2.88,
      "b365d": 3.25,
      "b365a": 2.50
    }
  ]
}
```

---

### `POST /api/v1/predict`

Predicts the outcome of a single match.

**Request body:**
```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "home_odd": 2.10,
  "draw_odd": 3.40,
  "away_odd": 3.20
}
```
`home_odd`, `draw_odd`, `away_odd` are optional. Without them, `ev_analysis` returns `null`.

**Response:**
```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "probabilities": { "H": 0.467, "D": 0.295, "A": 0.238 },
  "predicted": "H",
  "label": "Home Win",
  "confidence": 0.467,
  "ev_analysis": { ... }
}
```

---

### `GET /api/v1/predict/upcoming`

Returns predictions for all upcoming EPL fixtures with live B365 odds.

**Query params:** `limit` (default 20)

**Response:** Array of fixture prediction objects, each containing `fixture_id`, `date`, `home_team`, `away_team`, `b365`, `probabilities`, `predicted`, `label`, `confidence`, and `ev_analysis`.

---

## Project Structure

```
epl-predictor/                    ← Next.js frontend
├── src/
│   ├── app/
│   │   ├── upcoming/page.tsx
│   │   ├── predict/page.tsx
│   │   └── results/page.tsx
│   ├── components/
│   │   ├── match/
│   │   │   ├── MatchCard.tsx
│   │   │   ├── PredictionBadge.tsx
│   │   │   ├── OddsRow.tsx
│   │   │   └── EVTable.tsx
│   │   ├── stats/
│   │   │   └── AccuracyStats.tsx
│   │   └── layout/
│   │       └── Navbar.tsx
│   ├── lib/
│   │   ├── api.ts
│   │   └── errors.ts
│   └── types/index.ts
└── package.json

epl_predictor_v2/                 ← FastAPI backend
├── app/
│   ├── api/v1/endpoints/
│   │   ├── predict.py
│   │   └── matches.py
│   ├── services/
│   │   ├── feature_service.py
│   │   ├── odds_service.py
│   │   └── ev_service.py
│   ├── schemas/
│   │   └── prediction.py
│   └── ml/features/
│       └── feature_columns.py
├── models/saved/
│   ├── ensemble.pkl
│   ├── imputer.pkl
│   └── feature_names.pkl
├── scripts/
│   └── train_model.py
├── data/processed/
│   └── features.parquet
├── Dockerfile
└── requirements.txt
```

---

## Setup & Running Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- A Supabase project with historical EPL match data loaded
- An OddsPapi API key

### Backend

```bash
cd epl_predictor_v2

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your Supabase URL, Supabase key, and OddsPapi key

# Train the model (first time only)
python scripts/train_model.py

# Start the API server
uvicorn app.main:app --reload --port 8000
```

API is now available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd epl-predictor

# Install dependencies
npm install

# Set environment variable
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```

Frontend is now available at `http://localhost:3000`.

---

## Results & Performance

### Model Accuracy

The ensemble model achieves the following on the held-out validation set (2024/25 season):

| Metric | Score |
|---|---|
| Overall accuracy | ~54–56% |
| Macro F1-score | ~0.50–0.52 |
| Home win precision | ~62% |
| Draw recall | ~38% (hardest class) |

These figures are consistent with published research on EPL outcome prediction. Draws are the hardest class to predict across all football prediction models due to their lower base rate and high variance.

### Value Bet Identification

The EV engine flags approximately 15–25% of fixtures as containing at least one value bet per gameweek, depending on market efficiency that week. The 5% EV threshold was calibrated on historical data to balance bet frequency against edge quality.

### Limitations

- **Injury and suspension data** are not currently incorporated. A key player absence can dramatically shift outcome probabilities in ways the rolling stats model cannot capture.
- **Home advantage** is partially encoded via the home/away feature split but is not explicitly modelled as a time-varying effect (e.g., post-COVID matches played behind closed doors).
- **Hugging Face free tier sleeping** means the backend may have a 30–60 second cold start on the first request after inactivity.
- **Odds availability** depends on OddsPapi coverage. Fixtures more than two weeks out may not have B365 odds yet, causing them to be skipped in the upcoming endpoint.
