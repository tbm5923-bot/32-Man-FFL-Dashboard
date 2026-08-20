# 32-Man FFL Dashboard — MIDA

Interactive Streamlit dashboard for the **McAllister Institute for Dynasty Analytics (MIDA)** 32-team dynasty forecasting model.

This repository is the **web/dashboard layer only**. It reads a frozen forecast snapshot from `outputs/` and does not run or modify the forecasting engine.

## Current snapshot

- Engine: MIDA v0.6.2
- Season: 2026 preseason
- Simulations: 20,000
- Regular season: Weeks 1–13
- Playoffs: Weeks 14–17

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

## Deploy on Streamlit Community Cloud

Use:

- Repository: `tbm5923-bot/32-Man-FFL-Dashboard`
- Branch: `streamlit-deploy-v1`
- Main file path: `dashboard/app.py`

The app reads the checked-in files in `outputs/`.

## Updating the public forecast

1. Run the MIDA forecasting engine separately on the owner machine.
2. Replace the canonical files in `outputs/` with the new forecast bundle.
3. Commit the updated output snapshot to this repository.
4. Streamlit Community Cloud redeploys from the updated repository.

## Dashboard views

- League
- Team
- Matchups
- Players
- Playoffs
- Model Lab
- Engine Health

## Architecture

```text
MIDA forecasting engine (private/local)
             |
             v
        output CSV/JSON
             |
             v
   this read-only repository
             |
             v
     Streamlit web dashboard
```

### Deployment data note

The frozen web forecast is stored as a compact, losslessly compressed snapshot split into GitHub-safe text chunks. The dashboard loader reconstructs it in memory; the underlying MIDA engine outputs remain unchanged.
