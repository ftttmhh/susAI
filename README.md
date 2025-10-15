# Sustainable AI Model Recommender

## Backend Setup

1. Open a terminal in `backend/`:

```
cd backend
python -m venv .venv
.venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

2. Place your `model_metadata.csv` in the backend folder.

3. Run the backend:

```
uvicorn main:app --reload --port 8000
```

## Frontend Setup

1. Open a terminal in `frontend/`:

```
cd frontend
npm install
npm start
```

2. Open [http://localhost:3000](http://localhost:3000) in your browser.

--------------------------OR-----------------------------------------
## Start both frontend & backend quickly (Windows)

Use the provided `start-dev.bat` from the project root to open two terminals and start both services automatically (backend with the virtualenv activation and uvicorn, frontend with `npm start`). From PowerShell or Explorer run:

```powershell
# from project root
.\start-dev.bat
```

This will open two new command windows:
- Backend: activates `backend/.venv` and runs `uvicorn main:app --reload --port 8000`
- Frontend: runs `npm start` in the `frontend` folder

If you prefer a single-window approach you can install `concurrently` and run both services in one terminal (see notes in repo). The `start-dev.bat` approach is the recommended quick dev workflow on Windows since it uses cmd activation for the venv.

-
-
-
-
-

HELP YOURSELF!
## Notes
- The backend uses BERT for task classification and filters models from the CSV.
- The frontend provides a UI for task input, constraints, and displays recommendations.
- For production, improve security, error handling, and performance as needed.

## Baseline fields explained

- `baseline_input`: the text the user entered or selected as their go-to model.
- `baseline_matched_model`: the exact model row from `model_metadata.csv` that the system matched to the user's input.
- `baseline_energy_wh_per_1k`: the GPU energy (Wh per 1k inferences) for the matched baseline model, if available.
- `baseline_in_task`: true if the matched baseline row belongs to the same task category used for recommendations. When false, the baseline is not considered for comparisons to avoid cross-task apples-to-oranges comparisons.
- `baseline_passed_filters`: true if the matched baseline passes the user's hard constraints (for example minimum accuracy or carbon budget). If false, the baseline was filtered out before scoring, so it cannot be recommended.
- `baseline_comparable_score`: a composite score computed by normalizing metrics across the union of the candidate models and the baseline so you can compare them on the same scale (this is only computed when the baseline belongs to the same task). It uses the same weights and normalization procedure that the recommender uses for ranking.

Notes on interpretation:
- `energy_saved_wh_per_1k` displayed in the results is computed as `max(baseline_energy - candidate_energy, 0)`. It is NOT a sum — it is the amount of energy (Wh per 1k inferences) you would save by switching from the baseline to the candidate.
- `baseline_comparable_score` is the recomputed score for the baseline when included in the same normalized pool; compare it to the `score` values in the recommendations to see why a baseline was or wasn't ranked higher.


