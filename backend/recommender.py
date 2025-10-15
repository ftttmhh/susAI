# backend/recommender.py
import pandas as pd
import numpy as np
import difflib

def normalize(series):
    """Min-max normalization, handle constant values and NaN values."""
    # Remove NaN values for normalization
    clean_series = series.dropna().astype(float)
    out = pd.Series(index=series.index, dtype=float)
    if clean_series.empty:
        out[:] = 0.0
        return out
    if clean_series.max() == clean_series.min():
        # If all values equal, treat them as best (1.0)
        out[:] = 0.0
        out[clean_series.index] = 1.0
        return out

    normalized = (clean_series - clean_series.min()) / (clean_series.max() - clean_series.min())
    out[:] = 0.0
    out.loc[normalized.index] = normalized.astype(float)
    return out

def recommend(
    df,
    task,
    min_acc=None,
    max_lat=None,
    carbon_budget=None,
    priority="balanced",
    w_acc=None, w_energy=None, w_carbon=None, w_lat=None,
    topk=3,
    baseline_model=None
):
    print(f"[DEBUG] Starting recommendation for task: {task}")
    print(f"[DEBUG] Initial dataset size: {len(df)} rows")
    
    # 1. Filter by task
    # be defensive when task column contains NaN
    filtered = df[df['task'].fillna("").str.lower() == str(task).lower()].copy()
    print(f"[DEBUG] After task filter '{task}': {len(filtered)} rows")
    if filtered.empty:
        return [], {"note": f"No models found for task '{task}'."}

    # 2. Apply hard constraints (ignore NaN values)
    if min_acc is not None:
        before = len(filtered)
        filtered = filtered[filtered['accuracy'].notna() & (filtered['accuracy'] >= float(min_acc))]
        print(f"[DEBUG] After accuracy filter >= {min_acc}: {len(filtered)} rows (filtered out {before - len(filtered)})")
    
    # Commented out latency filtering - will add benchmarked values later
    # if max_lat is not None:
    #     before = len(filtered)
    #     filtered = filtered[filtered['latency_ms'].notna() & (filtered['latency_ms'] <= max_lat)]
    #     print(f"[DEBUG] After latency filter <= {max_lat}: {len(filtered)} rows (filtered out {before - len(filtered)})")
    
    if carbon_budget is not None:
        before = len(filtered)
        filtered = filtered[filtered['co2_kg_per_1k'].notna() & (filtered['co2_kg_per_1k'] <= float(carbon_budget))]
        print(f"[DEBUG] After carbon filter <= {carbon_budget}: {len(filtered)} rows (filtered out {before - len(filtered)})")

    if filtered.empty:
        return pd.DataFrame(), {"note": "No models matched after applying constraints."}

    # 3. Normalize metrics (only for non-NaN values)
    print(f"[DEBUG] Normalizing metrics for {len(filtered)} models")
    filtered = filtered.copy()
    filtered["acc_norm"] = normalize(filtered.get("accuracy", pd.Series(dtype=float)))
    # Commented out latency normalization - will add benchmarked values later
    # filtered["lat_norm"] = 1 - normalize(filtered["latency_ms"])  # lower latency is better
    filtered["energy_norm"] = 1 - normalize(filtered.get("gpu_energy", pd.Series(dtype=float)))  # lower energy is better
    filtered["carbon_norm"] = 1 - normalize(filtered.get("co2_kg_per_1k", pd.Series(dtype=float)))  # lower carbon is better

    # 4. Assign weights (adjusted for missing latency)
    if priority == "accuracy-first":
        weights = {"acc": 0.7, "energy": 0.15, "carbon": 0.15}  # removed lat weight
    elif priority == "green-first":
        weights = {"acc": 0.2, "energy": 0.4, "carbon": 0.4}  # removed lat weight
    else:  # balanced
        weights = {"acc": 0.5, "energy": 0.25, "carbon": 0.25}  # removed lat weight

    # Override if manual weights provided
    if w_acc is not None: weights["acc"] = float(w_acc)
    if w_lat is not None: weights["lat"] = float(w_lat)
    if w_energy is not None: weights["energy"] = float(w_energy)
    if w_carbon is not None: weights["carbon"] = float(w_carbon)

    # 5. Compute composite score (adjusted for missing latency)
    # compute weighted score; missing norms produce NaN, so fillna(0.0)
    filtered["score"] = (
        weights["acc"] * filtered["acc_norm"].fillna(0.0) +
        weights.get("lat", 0.0) * filtered.get("lat_norm", pd.Series(0.0, index=filtered.index)).fillna(0.0) +
        weights["energy"] * filtered["energy_norm"].fillna(0.0) +
        weights["carbon"] * filtered["carbon_norm"].fillna(0.0)
    )

    # 6. Sort by score
    ranked = filtered.sort_values("score", ascending=False).head(int(topk))
    print(f"[DEBUG] Final ranked models: {len(ranked)} rows")

    # 7. Clean up NaN values before returning (replace with 0 or appropriate defaults)
    for col in ranked.columns:
        if pd.api.types.is_float_dtype(ranked[col]) or pd.api.types.is_integer_dtype(ranked[col]):
            ranked[col] = ranked[col].fillna(0.0).astype(float)
        else:
            ranked[col] = ranked[col].fillna("N/A")

    # 8. Compute baseline comparison if given
    ctx = {"note": f"Ranked {len(ranked)} models for task '{task}'."}

    # Baseline matching: try exact match first, then case-insensitive substring in model or provider
    # If those fail, fall back to fuzzy matching using difflib.get_close_matches
    matched_baseline = None
    if baseline_model:
        baseline_str = str(baseline_model).strip().lower()
        # helper that prefers rows whose task matches the requested task
        def pick_preferred(df_rows):
            if df_rows.shape[0] == 0:
                return None
            task_lower = str(task).strip().lower()
            same_task = df_rows[df_rows["task"].fillna("").str.lower() == task_lower]
            if same_task.shape[0] > 0:
                return same_task.iloc[0]
            return df_rows.iloc[0]

        # exact match on model name
        exact_rows = df[df["model"].fillna("").str.strip().str.lower() == baseline_str]
        if exact_rows.shape[0] > 0:
            matched_baseline = pick_preferred(exact_rows)
            ctx["baseline"] = matched_baseline["model"]
        else:
            # substring match against model names
            candidates = df[df["model"].fillna("").str.lower().str.contains(baseline_str)]
            if candidates.shape[0] == 0:
                # try provider field
                candidates = df[df["provider"].fillna("").str.lower().str.contains(baseline_str)]
            if candidates.shape[0] > 0:
                matched_baseline = pick_preferred(candidates)
                ctx["baseline"] = matched_baseline["model"]

            # fuzzy match fallback
            if matched_baseline is None:
                model_names = [str(x) for x in df["model"].fillna("").values]
                # use a low cutoff to allow approximate matches
                close = difflib.get_close_matches(baseline_str, model_names, n=5, cutoff=0.4)
                if close:
                    # try to pick a close match whose task equals requested
                    close_rows = df[df["model"].fillna("").isin(close)]
                    if close_rows.shape[0] > 0:
                        matched_baseline = pick_preferred(close_rows)
                        ctx["baseline"] = matched_baseline["model"]

    if matched_baseline is not None:
        # record which baseline string was matched to which model
        ctx["baseline_input"] = baseline_model
        ctx["baseline_matched_model"] = matched_baseline.get("model")
        # baseline raw metrics
        try:
            b_acc = float(matched_baseline.get("accuracy", np.nan)) if not pd.isna(matched_baseline.get("accuracy", np.nan)) else None
        except Exception:
            b_acc = None
        try:
            b_energy = float(matched_baseline.get("gpu_energy", np.nan)) if not pd.isna(matched_baseline.get("gpu_energy", np.nan)) else None
        except Exception:
            b_energy = None
        try:
            b_co2 = float(matched_baseline.get("co2_kg_per_1k", np.nan)) if not pd.isna(matched_baseline.get("co2_kg_per_1k", np.nan)) else None
        except Exception:
            b_co2 = None
        ctx["baseline_raw"] = {"accuracy": b_acc, "gpu_energy": b_energy, "co2_kg_per_1k": b_co2}

        # Was baseline part of the same task set we filtered on?
        baseline_task = str(matched_baseline.get("task", "")).strip().lower()
        ctx["baseline_in_task"] = (baseline_task == str(task).strip().lower())

        # Did baseline pass the same hard filters?
        passed = True
        if b_acc is None and (min_acc is not None):
            passed = False
        if min_acc is not None and b_acc is not None and b_acc < float(min_acc):
            passed = False
        if carbon_budget is not None and b_co2 is not None and b_co2 > float(carbon_budget):
            passed = False
        ctx["baseline_passed_filters"] = passed

        # If baseline is part of the same task, compute a comparable score by normalizing across
        # the union of candidates + baseline so we can compare apples-to-apples.
        if ctx["baseline_in_task"]:
            try:
                # create a union DataFrame containing the ranked candidates and the baseline row
                base_df = pd.DataFrame([matched_baseline])
                union = pd.concat([filtered, base_df], ignore_index=True, sort=False)
                # recompute norms on union
                union_acc_norm = normalize(union.get("accuracy", pd.Series(dtype=float)))
                union_energy_norm = 1 - normalize(union.get("gpu_energy", pd.Series(dtype=float)))
                union_carbon_norm = 1 - normalize(union.get("co2_kg_per_1k", pd.Series(dtype=float)))
                # weights used above
                w_acc = weights["acc"]
                w_energy = weights["energy"]
                w_carbon = weights["carbon"]
                union_score = (w_acc * union_acc_norm.fillna(0.0) + w_energy * union_energy_norm.fillna(0.0) + w_carbon * union_carbon_norm.fillna(0.0))
                # baseline score is last row in union (we appended it)
                baseline_score = float(union_score.iloc[-1])
                ctx["baseline_comparable_score"] = baseline_score
            except Exception:
                ctx["baseline_comparable_score"] = None
        if "gpu_energy" in matched_baseline and not pd.isna(matched_baseline["gpu_energy"]):
            base_energy = float(matched_baseline["gpu_energy"])
            ctx["baseline_energy_wh_per_1k"] = base_energy
            if "gpu_energy" in ranked.columns:
                # signed delta: candidate_energy - baseline_energy (positive means candidate uses MORE energy)
                ranked["energy_delta_wh_per_1k"] = (ranked["gpu_energy"] - base_energy).fillna(0.0)
                # energy_saved is positive only when candidate uses less energy than baseline
                ranked["energy_saved_wh_per_1k"] = (base_energy - ranked["gpu_energy"]).clip(lower=0.0).fillna(0.0)
            else:
                ranked["energy_delta_wh_per_1k"] = 0.0
                ranked["energy_saved_wh_per_1k"] = 0.0
        else:
            # matched baseline but it lacks gpu_energy info
            ranked["energy_delta_wh_per_1k"] = 0.0
            ranked["energy_saved_wh_per_1k"] = 0.0
            ctx["baseline_note"] = "Matched baseline model lacks gpu_energy data; cannot compute savings."
    else:
        # baseline not provided or not matched — zero savings
        ranked["energy_delta_wh_per_1k"] = 0.0
        ranked["energy_saved_wh_per_1k"] = 0.0
        if baseline_model:
            ctx["baseline_note"] = f"No model matched for baseline input '{baseline_model}'. Use an exact model name from metadata or try a different name."

    # 9. Return
    # Convert to records for safe JSON serialization by the API layer
    return ranked.reset_index(drop=True), ctx
