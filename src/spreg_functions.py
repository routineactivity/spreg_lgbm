# Modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# Shared Utilities
def _select_target(df: pd.DataFrame, use_crime_count: bool = True) -> str:
    """
    Choose a single target column. If use_crime_count=True, prefer 'crime_count'.
    Otherwise prefer 'crime_density' then 'crime_kde'.
    """
    if use_crime_count:
        if "crime_count" in df.columns:
            return "crime_count"
        # fallbacks if missing
        for c in ("crime_density", "crime_kde"):
            if c in df.columns:
                return c
        raise ValueError("No target found. Expected one of: crime_count, crime_density, crime_kde.")
    else:
        for c in ("crime_density", "crime_kde"):
            if c in df.columns:
                return c
        # fallback to count if the others are missing (explicit)
        if "crime_count" in df.columns:
            return "crime_count"
        raise ValueError("No target found. Expected one of: crime_density, crime_kde, crime_count.")

def _get_id_col(df: pd.DataFrame):
    cands = [c for c in df.columns if c.lower() in ("unit_id","unitid","id","segment_id","hex_id")]
    return cands[0] if cands else None

def _metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

def _narrative_top_positive(feature_summary: pd.DataFrame, k: int = 5) -> str:
    fs = feature_summary.sort_values("mean_positive_contrib", ascending=False).head(k)
    lines = []
    for _, r in fs.iterrows():
        lines.append(
            f"- Higher **{r['feature']}** tended to increase predicted crime "
            f"(avg +contrib ≈ {r['mean_positive_contrib']:.3f})."
        )
    return "\n".join(lines)

# Fit Model and Contribs
def fit_lightgbm_kde_dist(
    df: pd.DataFrame,
    use_crime_count: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    lgb_params: dict | None = None,
) -> dict:
    """
    Train LightGBM on KDE_ and Dist_ features to predict a single target.
    Returns all artifacts in memory (no files). Prints top positive drivers (plain language).
    """
    # Features
    kde_cols  = [c for c in df.columns if c.startswith("KDE_")]
    dist_cols = [c for c in df.columns if c.startswith("Dist_")]
    feature_cols = kde_cols + dist_cols
    if not feature_cols:
        raise ValueError("No features found. Expect columns starting with KDE_ and/or Dist_.")
    
    # Target
    target_col = _select_target(df, use_crime_count=use_crime_count)
    y_all = df[target_col]
    X_all = df[feature_cols]
    mask = y_all.notna()
    X = X_all.loc[mask]
    y = y_all.loc[mask]
    index_kept = X.index

    # Optional ID
    id_col = _get_id_col(df)

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # LightGBM params
    if lgb_params is None:
        lgb_params = dict(
            n_estimators=1000,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=-1,
            n_jobs=-1,
            reg_lambda=1.0,
            random_state=random_state,
        )

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # Holdout metrics
    y_pred = model.predict(X_te)
    metrics = _metrics(y_te, y_pred)

    # TreeSHAP contributions (global + per-unit on ALL rows used for training)
    contrib_all = model.predict(X, pred_contrib=True)  # shape: (n, p+1) last col = bias
    feat_contrib = contrib_all[:, :-1]
    bias = contrib_all[:, -1]
    pred = model.predict(X)

    # Global feature summary
    mean_abs = np.mean(np.abs(feat_contrib), axis=0)
    mean_pos = np.where(feat_contrib > 0, feat_contrib, 0).mean(axis=0)
    mean_neg = np.where(feat_contrib < 0, feat_contrib, 0).mean(axis=0)
    mean_con = np.mean(feat_contrib, axis=0)

    feature_summary = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_contrib": mean_abs,
        "mean_positive_contrib": mean_pos,
        "mean_negative_contrib": mean_neg,
        "mean_contrib": mean_con,
        "gain_importance": model.booster_.feature_importance(importance_type="gain"),
        "split_importance": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("mean_abs_contrib", ascending=False).reset_index(drop=True)

    # Per-unit contributions matrix (in memory)
    contrib_df = pd.DataFrame(feat_contrib, columns=feature_cols, index=index_kept)
    contrib_df.insert(0, "bias", bias)
    contrib_df.insert(0, "prediction", pred)
    if id_col:
        contrib_df.insert(0, id_col, df.loc[index_kept, id_col].values)

    # Plain-language bullets (printed)
    print(f"Holdout metrics for {target_col}: {metrics}")
    print("\nTop positive drivers (plain language):")
    print(_narrative_top_positive(feature_summary, k=5))

    return {
        "target_col": target_col,
        "feature_cols": feature_cols,
        "model": model,
        "metrics": metrics,
        "feature_summary": feature_summary,   # DataFrame
        "contrib_df": contrib_df,            # DataFrame (per-unit)
        "id_col": id_col,
        "index_used": index_kept,            # original index subset
        "kde_cols": kde_cols,
        "dist_cols": dist_cols,
    }

# Validate
def validate_lightgbm_model(
    df: pd.DataFrame,
    use_crime_count: bool = True,
    k_folds: int = 5,
    random_state: int = 42,
    lgb_params: dict | None = None,
) -> pd.DataFrame:
    """
    K-fold CV (random folds). Returns a metrics DataFrame (in memory).
    For spatial CV, replace KFold with GroupKFold and pass a grouping key.
    """
    # Features and target selection identical to Task 1
    kde_cols  = [c for c in df.columns if c.startswith("KDE_")]
    dist_cols = [c for c in df.columns if c.startswith("Dist_")]
    feature_cols = kde_cols + dist_cols
    if not feature_cols:
        raise ValueError("No features found. Expect KDE_* and/or Dist_*.")

    tgt = _select_target(df, use_crime_count=use_crime_count)
    y_all = df[tgt]
    X_all = df[feature_cols]
    mask = y_all.notna()
    X = X_all.loc[mask]
    y = y_all.loc[mask]

    if lgb_params is None:
        lgb_params = dict(
            n_estimators=800,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=-1,
            n_jobs=-1,
            reg_lambda=1.0,
            random_state=random_state,
        )

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    rows = []
    for i, (tr, va) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        pred = model.predict(X_va)
        m = _metrics(y_va, pred)
        m["fold"] = i
        rows.append(m)

    cv_df = pd.DataFrame(rows)
    cv_df.loc["mean"] = {
        "MAE": cv_df["MAE"].mean(),
        "RMSE": cv_df["RMSE"].mean(),
        "R2": cv_df["R2"].mean(),
        "fold": "mean",
    }
    return cv_df

# Explainers
def explain_and_prepare_contribs(
    model_out: dict,
    top_k_global: int = 10,
    unit_row: int | None = None,
    make_zscore: bool = True
) -> dict:
    """
    Takes output from fit_lightgbm_kde_dist, prints narrative bullets and top positive pushes,
    and returns in-memory matrices: all, positive_only, negative_only, and z-scored (optional).
    """
    feature_summary: pd.DataFrame = model_out["feature_summary"]
    contrib_df: pd.DataFrame = model_out["contrib_df"].copy()
    id_col = model_out.get("id_col")

    # 1) Narrative bullets (global)
    print("Narrative — top positive global drivers:")
    print(_narrative_top_positive(feature_summary, k=min(top_k_global, len(feature_summary))))

    # 2) Global top positive pushes table (print)
    top_pos = feature_summary.sort_values("mean_positive_contrib", ascending=False).head(top_k_global)
    display(top_pos[["feature","mean_positive_contrib","mean_abs_contrib","gain_importance","split_importance"]])

    # 3) Optional local "top pushes" for a unit
    if unit_row is not None:
        cols = [c for c in contrib_df.columns if c not in ("prediction","bias", id_col)]
        s = contrib_df.iloc[unit_row][cols]
        unit_label = contrib_df.iloc[unit_row][id_col] if id_col else unit_row
        print(f"\nLocal explanation — unit: {unit_label} | "
              f"Prediction={contrib_df.iloc[unit_row]['prediction']:.3f} | "
              f"Bias={contrib_df.iloc[unit_row]['bias']:.3f}")
        print("Top positive pushes:")
        display(s.sort_values(ascending=False).head(10).to_frame("contribution"))
        print("Top negative pushes:")
        display(s.sort_values(ascending=True).head(10).to_frame("contribution"))

    # 4) Matrices for clustering
    cols_feat = [c for c in contrib_df.columns if c not in ("prediction","bias", id_col)]
    contrib_all = contrib_df.copy()

    contrib_pos = contrib_df[[id_col] + cols_feat].copy() if id_col else contrib_df[cols_feat].copy()
    contrib_neg = contrib_pos.copy()
    for c in cols_feat:
        contrib_pos[c] = contrib_pos[c].clip(lower=0)
        contrib_neg[c] = contrib_neg[c].clip(upper=0)

    contrib_z = None
    if make_zscore:
        M = contrib_df[cols_feat].values
        mu = M.mean(axis=0)
        sd = M.std(axis=0) + 1e-9
        Mz = (M - mu) / sd
        contrib_z = pd.DataFrame(Mz, columns=[f"z_{c}" for c in cols_feat], index=contrib_df.index)
        if id_col:
            contrib_z.insert(0, id_col, contrib_df[id_col].values)

    # Return everything in memory
    return {
        "top_positive_table": top_pos,   # DataFrame
        "contrib_all": contrib_all,      # DataFrame (prediction, bias, features)
        "contrib_positive_only": contrib_pos,
        "contrib_negative_only": contrib_neg,
        "contrib_zscore": contrib_z,     # DataFrame or None
    }

