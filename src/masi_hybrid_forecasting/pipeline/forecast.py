"""
forecast command — projection MASI hors-échantillon (mai/juin ou autre horizon).

Pipeline hybride :
  1. base ARIMA sur les rendements log historiques (sélection AIC parmi 8 ordres)
  2. simulation Monte-Carlo (N trajectoires) -> bandes p10/p50/p90 robustes
  3. conditionnement régime HMM : si le dernier régime est Bear/Bull, on
     décale la dérive par le rendement moyen historique du régime
  4. sortie : panel journalier + résumé mensuel + markdown (optionnel)

Si `outputs/etape6/etape6_final_predictions.csv` couvre déjà la période cible,
ces lignes sont réutilisées telles-quelles (source="pipeline") au lieu de la
projection ARIMA — c'est le cas dès qu'on relance les étapes 01->06 sur des
données réelles fraîches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CANONICAL_CSV, OUTPUTS_DIR, PROJECT_ROOT, REGIMES_CSV

logger = logging.getLogger(__name__)

HISTORY_CSV = OUTPUTS_DIR / "etape1" / "splits" / "masi_clean_full.csv"
PRICE_COL = "masi_close"

CANDIDATE_ORDERS = [
    (0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1),
    (2, 0, 0), (0, 0, 2), (2, 0, 1), (1, 0, 2),
]
Z80 = 1.2815515655446004   # quantile normal pour bandes 80 %
Z95 = 1.959963984540054    # quantile normal pour bandes 95 %


# ============================================================================
# DATACLASS RÉSULTAT
# ============================================================================
@dataclass
class ForecastResult:
    daily: pd.DataFrame          # date, source, predicted_log_return, predicted_close, p10/p50/p90, signal
    monthly: pd.DataFrame        # month, n_days, start/end close, monthly_return_pct, ...
    arima_order: tuple | None
    arima_aic: float | None
    regime_drift: float          # décalage de drift appliqué (log-return / jour)
    n_mc: int
    last_obs_date: pd.Timestamp
    last_obs_close: float


# ============================================================================
# UTILITAIRES
# ============================================================================
def load_history(path: Path = HISTORY_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Historique introuvable : {path}\n"
            f"Lance d'abord `python scripts/01_preprocessing.py`."
        )
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").set_index("date")
    if PRICE_COL not in df.columns:
        raise KeyError(f"Colonne {PRICE_COL!r} absente de {path}")
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df[PRICE_COL]).diff()
    return df[[PRICE_COL, "log_return"]].dropna().copy()


def load_pipeline_window(year: int, months: list[int]) -> pd.DataFrame:
    """Renvoie les prédictions pipeline étape 6 sur la fenêtre cible (peut être vide)."""
    if not CANONICAL_CSV.exists():
        return pd.DataFrame()
    preds = pd.read_csv(CANONICAL_CSV, parse_dates=["date"])
    mask = (preds["date"].dt.year == year) & (preds["date"].dt.month.isin(months))
    return preds.loc[mask].copy()


def estimate_regime_drift(regimes_path: Path = REGIMES_CSV,
                          history: pd.DataFrame | None = None) -> tuple[float, str]:
    """
    Renvoie (drift, label) où drift est le rendement log moyen historique du
    dernier régime HMM observé. Si Neutral ou indisponible -> drift = 0.
    """
    if not regimes_path.exists() or history is None:
        return 0.0, "neutral_default"
    regs = pd.read_csv(regimes_path, parse_dates=["date"]).sort_values("date")
    if regs.empty:
        return 0.0, "neutral_default"
    last = regs.iloc[-1]
    label = str(last.get("regime_name", "Neutral"))
    if label == "Neutral":
        return 0.0, label
    merged = regs.merge(history.reset_index(), on="date", how="inner")
    mask = merged["regime_name"] == label
    if not mask.any():
        return 0.0, label
    return float(merged.loc[mask, "log_return"].mean()), label


# ============================================================================
# CŒUR : ARIMA + MONTE-CARLO
# ============================================================================
def fit_best_arima(returns: pd.Series):
    """Sélection AIC sur 8 ordres candidats."""
    from statsmodels.tsa.arima.model import ARIMA
    best, best_aic, best_order = None, np.inf, None
    for order in CANDIDATE_ORDERS:
        try:
            fit = ARIMA(returns, order=order, trend="c",
                        enforce_stationarity=False, enforce_invertibility=False).fit()
        except Exception:                                       # noqa: BLE001
            continue
        if fit.aic < best_aic:
            best, best_aic, best_order = fit, float(fit.aic), order
    if best is None:
        raise RuntimeError("Aucun modèle ARIMA n'a convergé sur l'historique.")
    return best, best_order, best_aic


def simulate_paths(mean_returns: np.ndarray, sigma: float, n_sim: int,
                   horizon: int, last_price: float, drift_shift: float = 0.0,
                   seed: int = 1234) -> np.ndarray:
    """
    Monte-Carlo simple : autour de la prévision conditionnelle ARIMA on ajoute
    des chocs gaussiens iid d'écart-type `sigma` (résidu ARIMA).
    Retourne un array (n_sim, horizon) de prix simulés.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=0.0, scale=sigma, size=(n_sim, horizon))
    paths_log = (mean_returns + drift_shift) + shocks            # broadcasting
    cum = np.cumsum(paths_log, axis=1)
    return last_price * np.exp(cum)


# ============================================================================
# API HAUT-NIVEAU
# ============================================================================
def run_forecast(target_year: int,
                 target_months: list[int],
                 n_mc: int = 5_000,
                 use_regime_drift: bool = True,
                 seed: int = 1234) -> ForecastResult:
    history = load_history()
    last_date = history.index.max()
    last_price = float(history[PRICE_COL].iloc[-1])

    target_start = pd.Timestamp(year=target_year, month=min(target_months), day=1)
    target_end = pd.Timestamp(year=target_year, month=max(target_months), day=1) + pd.offsets.MonthEnd(0)

    # 1. ARIMA + drift régime
    forecast_start = last_date + pd.offsets.BDay(1)
    forecast_dates = pd.bdate_range(forecast_start, target_end)
    drift_shift, regime_label = (
        estimate_regime_drift(history=history) if use_regime_drift else (0.0, "disabled")
    )

    best_fit, best_order, best_aic = None, None, None
    fc_returns = np.array([])
    sigma = np.nan

    if len(forecast_dates) > 0:
        best_fit, best_order, best_aic = fit_best_arima(history["log_return"])
        fc = best_fit.get_forecast(steps=len(forecast_dates))
        fc_returns = np.asarray(fc.predicted_mean, dtype=float) + drift_shift
        sigma = float(np.nanstd(best_fit.resid, ddof=1))

        # 2. Monte-Carlo
        mc_paths = simulate_paths(
            mean_returns=np.asarray(fc.predicted_mean, dtype=float),
            sigma=sigma, n_sim=n_mc, horizon=len(forecast_dates),
            last_price=last_price, drift_shift=drift_shift, seed=seed,
        )
        p10 = np.percentile(mc_paths, 10, axis=0)
        p50 = np.percentile(mc_paths, 50, axis=0)
        p90 = np.percentile(mc_paths, 90, axis=0)
        cumulative = np.cumsum(fc_returns)
        steps = np.arange(1, len(forecast_dates) + 1)
        # bande analytique 95 % (sanity check / complément)
        p025 = last_price * np.exp(cumulative - Z95 * sigma * np.sqrt(steps))
        p975 = last_price * np.exp(cumulative + Z95 * sigma * np.sqrt(steps))

        future = pd.DataFrame({
            "date": forecast_dates,
            "predicted_log_return": fc_returns,
            "predicted_close": last_price * np.exp(cumulative),
            "mc_p10": p10, "mc_p50": p50, "mc_p90": p90,
            "ci95_low": p025, "ci95_high": p975,
        })
    else:
        future = pd.DataFrame(columns=[
            "date", "predicted_log_return", "predicted_close",
            "mc_p10", "mc_p50", "mc_p90", "ci95_low", "ci95_high",
        ])

    # 3. Réutilisation pipeline (si dispo)
    pipe = load_pipeline_window(target_year, target_months)
    pipe_panel = pd.DataFrame()
    if not pipe.empty:
        pipe_panel = pd.DataFrame({
            "date": pipe["date"].values,
            "source": "pipeline",
            "predicted_log_return": pipe["predicted_return"].values,
            "predicted_close": np.nan,        # le pipeline donne le rendement, pas le prix futur
        })

    # 4. Observed dans la fenêtre (au cas où l'historique recouvre déjà mai/juin)
    obs = history.loc[(history.index >= target_start) & (history.index <= target_end)]
    obs_panel = pd.DataFrame({
        "date": obs.index,
        "source": "observed",
        "predicted_log_return": obs["log_return"].values,
        "predicted_close": obs[PRICE_COL].values,
    })

    # 5. Forecast restreint à la fenêtre mai/juin
    fc_panel = future[(future["date"] >= target_start) & (future["date"] <= target_end)].copy()
    if not fc_panel.empty:
        fc_panel.insert(1, "source", "forecast")

    daily = pd.concat([obs_panel, pipe_panel, fc_panel], ignore_index=True)
    daily = daily.drop_duplicates(subset=["date"], keep="first").sort_values("date")
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    daily["signal"] = np.select(
        [daily["predicted_log_return"] > 0, daily["predicted_log_return"] < 0],
        ["hausse", "baisse"], default="neutre",
    )

    # 6. Résumé mensuel
    monthly_rows: list[dict] = []
    # chemin de prix combiné historique + futur pour le start-of-month
    price_path = pd.concat([
        history[[PRICE_COL]].rename(columns={PRICE_COL: "close"}),
        future.set_index("date")[["predicted_close"]].rename(columns={"predicted_close": "close"}),
    ]).sort_index()
    for m in target_months:
        m_start = pd.Timestamp(year=target_year, month=m, day=1)
        m_end = m_start + pd.offsets.MonthEnd(0)
        panel = daily[(daily["date"] >= m_start) & (daily["date"] <= m_end)]
        if panel.empty:
            continue
        prior = price_path[price_path.index < m_start]
        start_ref = float(prior["close"].iloc[-1]) if not prior.empty else float(panel["predicted_close"].dropna().iloc[0])
        end_close = float(panel["predicted_close"].dropna().iloc[-1]) if panel["predicted_close"].notna().any() else start_ref * float(np.exp(panel["predicted_log_return"].sum()))
        ret = float(np.log(end_close / start_ref))
        monthly_rows.append({
            "month": m_start.strftime("%Y-%m"),
            "source": "+".join(sorted(panel["source"].unique())),
            "n_days": int(len(panel)),
            "start_reference_close": start_ref,
            "end_predicted_close": end_close,
            "monthly_log_return": ret,
            "monthly_simple_return_pct": 100 * (np.exp(ret) - 1),
            "mean_daily_predicted_return": float(panel["predicted_log_return"].mean()),
            "positive_days": int((panel["predicted_log_return"] > 0).sum()),
            "negative_days": int((panel["predicted_log_return"] < 0).sum()),
        })
    monthly = pd.DataFrame(monthly_rows)

    return ForecastResult(
        daily=daily, monthly=monthly,
        arima_order=best_order, arima_aic=best_aic,
        regime_drift=drift_shift, n_mc=n_mc if not fc_panel.empty else 0,
        last_obs_date=last_date, last_obs_close=last_price,
    )


# ============================================================================
# EXPORT MARKDOWN
# ============================================================================
def write_report(result: ForecastResult, out_path: Path, target_year: int) -> None:
    lines = [
        f"# Projection MASI — {target_year} (mai + juin)",
        "",
        f"- Dernière observation : **{result.last_obs_date.date()}** (close {result.last_obs_close:,.2f})",
        f"- ARIMA retenu : `{result.arima_order}` (AIC {result.arima_aic:,.2f})" if result.arima_order else "- ARIMA : non utilisé (période déjà couverte)",
        f"- Drift régime HMM appliqué : {result.regime_drift:+.5f} log-return/jour",
        f"- Simulations Monte-Carlo : {result.n_mc:,}",
        "",
        "## Résumé mensuel",
        "",
        "| Mois | Source | Jours | Close début | Close fin | Rendement mensuel |",
        "|------|--------|------:|------------:|----------:|------------------:|",
    ]
    for row in result.monthly.to_dict("records"):
        lines.append(
            f"| {row['month']} | {row['source']} | {row['n_days']} | "
            f"{row['start_reference_close']:,.2f} | {row['end_predicted_close']:,.2f} | "
            f"{row['monthly_simple_return_pct']:+.2f} % |"
        )
    lines += [
        "",
        "## Lecture rapide",
        "",
    ]
    for row in result.monthly.to_dict("records"):
        direction = "hausse" if row["monthly_log_return"] > 0 else "baisse" if row["monthly_log_return"] < 0 else "neutre"
        lines.append(
            f"- **{row['month']}** : tendance **{direction}**, "
            f"rendement estimé {row['monthly_simple_return_pct']:+.2f} %, "
            f"close fin de mois ≈ {row['end_predicted_close']:,.2f}."
        )
    lines += [
        "",
        "> Projection statistique fondée sur l'historique disponible. "
        "Pour intégrer les vraies données mai/juin, relance les étapes 01 → 06 "
        "puis ré-exécute cette commande.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# CLI
# ============================================================================
def run(args) -> None:
    year = int(getattr(args, "year", 2026))
    months = [int(m) for m in getattr(args, "months", "5,6").split(",")]
    n_mc = int(getattr(args, "n_mc", 5_000))
    out_dir = Path(getattr(args, "output_dir", None) or (PROJECT_ROOT / "outputs" / "etape10"))
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run_forecast(target_year=year, target_months=months, n_mc=n_mc)

    daily_csv = out_dir / f"forecast_may_june_{year}_daily.csv"
    monthly_csv = out_dir / f"forecast_may_june_{year}_monthly.csv"
    report_md = out_dir / "report.md"
    res.daily.to_csv(daily_csv, index=False)
    res.monthly.to_csv(monthly_csv, index=False)
    write_report(res, report_md, year)

    logger.info(f"Forecast écrit : {daily_csv}")
    logger.info(f"Résumé mensuel  : {monthly_csv}")
    logger.info(f"Rapport         : {report_md}")
    logger.info(f"ARIMA {res.arima_order} AIC={res.arima_aic} | drift régime={res.regime_drift:+.5f}")
    for row in res.monthly.to_dict("records"):
        logger.info(
            f"  {row['month']} | {row['source']:>18} | "
            f"close fin ≈ {row['end_predicted_close']:>9,.2f} | "
            f"ret {row['monthly_simple_return_pct']:+.2f} %"
        )
