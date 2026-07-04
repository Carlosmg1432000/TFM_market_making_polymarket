"""
Motor de backtesting.

Version revisada:
- Analiza un unico token_id cada vez.
- Si no se indica token_id, selecciona automaticamente el token con mas dinamica.
- Incluye descriptiva previa de mercado.
- Guarda spread real observado y spread cotizado por estrategia.
- Ejecuta estrategias con probabilidad de ejecucion decreciente con la distancia al mid.
- Permite multiples simulaciones con semillas distintas.
- La estrategia naive usa como spread fijo la mediana del spread real observado.
- El spread fijo de la naive se acota para evitar que quede inactiva en mercados con spreads extremos.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .collector import load_data
from .signals import add_signals, zscore_diagnostics
from .strategies import (
    NaiveStrategy,
    HeuristicStrategy,
    BayesStrategy,
    BayesVPINStrategy,
)
from .config import DATA_DIR, SIGNAL_WINDOW, TOXIC_HORIZON_TICKS, TOXIC_MOVE_THRESHOLD


def max_drawdown(pnl):
    """Calcula el maximo drawdown de una trayectoria de PnL."""

    pnl = np.asarray(pnl, dtype=float)

    if len(pnl) == 0:
        return 0.0

    peak = np.maximum.accumulate(pnl)
    drawdown = pnl - peak

    return float(drawdown.min())


def sharpe(pnl):
    """Calcula un Sharpe simple no anualizado."""

    pnl = np.asarray(pnl, dtype=float)

    if len(pnl) < 3:
        return 0.0

    returns = np.diff(pnl)
    std = returns.std()

    if std < 1e-8:
        return 0.0

    return float(returns.mean() / std)


def _new_strategies(naive_spread=0.02):
    """
    Crea estrategias nuevas para cada simulacion.

    La naive recibe desde el backtester un spread fijo calibrado
    con la muestra de mercado analizada.
    """

    return [
        NaiveStrategy(fixed_spread=naive_spread),
        HeuristicStrategy(),
        BayesStrategy(),
        BayesVPINStrategy(),
    ]


def _safe_get(row, col, default=0.0):
    """Obtiene una columna de forma segura desde una fila de pandas."""

    if col in row.index:
        value = row[col]
        if pd.isna(value):
            return default
        return value

    return default


def _select_best_token(df, min_rows=250):
    """
    Selecciona automaticamente el token con mayor informacion dinamica.

    Criterio:
    - minimo numero de filas
    - mayor numero de cambios de midprice
    - mayor variabilidad de spread
    - mayor variabilidad de imbalance

    Esto evita ejecutar el backtest sobre mercados congelados.
    """

    scores = []

    for token_id, g in df.groupby("token_id"):
        g = g.sort_values("timestamp").copy()

        if len(g) < min_rows:
            continue

        mid_changes = int((g["midprice"].diff().fillna(0.0) != 0).sum())

        if "best_bid" in g.columns and "best_ask" in g.columns:
            market_spread = g["best_ask"] - g["best_bid"]
        else:
            market_spread = g["spread"]

        spread_unique = int(market_spread.nunique())
        imbalance_unique = int(g["imbalance"].nunique()) if "imbalance" in g.columns else 0

        mid_range = float(g["midprice"].max() - g["midprice"].min())
        spread_range = float(market_spread.max() - market_spread.min())

        score = (
            10 * mid_changes
            + 3 * spread_unique
            + 1 * imbalance_unique
            + 1000 * mid_range
            + 100 * spread_range
        )

        scores.append(
            {
                "token_id": token_id,
                "rows": len(g),
                "mid_changes": mid_changes,
                "mid_unique": int(g["midprice"].nunique()),
                "spread_unique": spread_unique,
                "imbalance_unique": imbalance_unique,
                "mid_range": mid_range,
                "spread_range": spread_range,
                "score": score,
            }
        )

    ranking = pd.DataFrame(scores)

    if ranking.empty:
        return None, ranking

    ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)

    return ranking.iloc[0]["token_id"], ranking


def market_descriptive_summary(df):
    """Genera una tabla descriptiva sencilla de los datos de mercado."""

    if df.empty:
        return pd.DataFrame()

    cols = [
        "midprice",
        "market_spread",
        "bid_depth",
        "ask_depth",
        "total_depth",
        "imbalance",
        "drift",
    ]

    available = [c for c in cols if c in df.columns]

    desc = (
        df[available]
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    return desc


def prepare_data(
    token_id=None,
    signal_window=SIGNAL_WINDOW,
    toxic_horizon=TOXIC_HORIZON_TICKS,
    toxic_move_threshold=TOXIC_MOVE_THRESHOLD,
):
    """
    Carga datos, evita mezclar mercados y anade senales.

    Si no se indica token_id, se selecciona automaticamente el token
    con mayor dinamica observada.
    """

    # Cargamos todos los datos para poder elegir token.
    df = load_data(token_id=None)

    if df.empty:
        print("No hay datos guardados. Primero captura order book.")
        return pd.DataFrame(), None

    if "token_id" not in df.columns:
        print("No existe columna token_id en los datos.")
        return pd.DataFrame(), None

    if token_id is None:
        selected_token, ranking = _select_best_token(df)

        if selected_token is None:
            print("No se ha encontrado ningun token con suficientes datos.")
            return pd.DataFrame(), None

        print("\nRANKING DE TOKENS POR DINAMICA OBSERVADA")

        ranking_cols = [
            "token_id",
            "rows",
            "mid_changes",
            "mid_unique",
            "spread_unique",
            "imbalance_unique",
            "score",
        ]

        print(
            ranking[ranking_cols]
            .head(10)
            .to_string(index=False)
        )

        print("\nToken seleccionado automaticamente:")
        print(selected_token)

        token_id = selected_token

    df = df[df["token_id"].astype(str) == str(token_id)].copy()

    if df.empty:
        print("No hay datos para el token_id indicado.")
        return pd.DataFrame(), None

    if len(df) < 30:
        print("Hay pocos datos. Captura al menos 100-300 snapshots.")
        return pd.DataFrame(), None

    df = df.sort_values("timestamp").reset_index(drop=True)

    df = add_signals(
        df,
        window=signal_window,
        toxic_horizon=toxic_horizon,
        toxic_move_threshold=toxic_move_threshold,
    )

    diagnostics = zscore_diagnostics(df)

    print("\nDESCRIPTIVA BASICA DEL MERCADO")
    print("Token:", token_id)
    print("Snapshots:", len(df))
    print("Midprice unique:", df["midprice"].nunique())
    print("Cambios de midprice:", int((df["midprice"].diff().fillna(0.0) != 0).sum()))
    print("Midprice medio:", round(float(df["midprice"].mean()), 6))
    print("Spread real medio:", round(float(df["market_spread"].mean()), 6))
    print("Spread real mediano:", round(float(df["market_spread"].median()), 6))
    print("Spread real unique:", df["market_spread"].nunique())
    print("Profundidad visible media:", round(float(df["total_depth"].mean()), 4))
    print("Imbalance medio:", round(float(df["imbalance"].mean()), 4))

    print("\nDIAGNOSTICO FLOW_Z_RAW")
    print("Media:", round(diagnostics["mean"], 4))
    print("Std:", round(diagnostics["std"], 4))
    print(
        "P05/P50/P95:",
        round(diagnostics["p05"], 4),
        round(diagnostics["p50"], 4),
        round(diagnostics["p95"], 4),
    )
    print("Share |z| > 3:", round(diagnostics["share_abs_gt_3"], 4))

    if "toxic_flow_label" in df.columns:
        print("Toxic flow rate:", round(float(df["toxic_flow_label"].mean()), 4))

    return df, token_id


def _summarize_paths(paths):
    """Calcula metricas resumen por estrategia para una trayectoria."""

    summary_rows = []

    for strategy_name, g in paths.groupby("strategy"):
        executions = int(g["executed_buy"].sum() + g["executed_sell"].sum())
        adverse_executions = int(g["adverse_event"].sum())

        adverse_event_rate = adverse_executions / executions if executions > 0 else 0.0

        summary_rows.append(
            {
                "strategy": strategy_name,
                "pnl_final": float(g["pnl"].iloc[-1]),
                "pnl_mean": float(g["pnl"].mean()),
                "pnl_volatility": float(g["pnl"].diff().std()),
                "max_drawdown": max_drawdown(g["pnl"]),
                "sharpe": sharpe(g["pnl"]),
                "inventory_abs_mean": float(g["inventory"].abs().mean()),
                "inventory_max_abs": float(g["inventory"].abs().max()),
                "spread_quoted_mean": float(g["spread"].mean()),
                "spread_market_mean": float(g["market_spread"].mean()),
                "signal_signed_mean": float(g["signal_signed"].mean()),
                "signal_signed_abs_mean": float(g["signal_signed"].abs().mean()),
                "defense_score_mean": float(g["defense_score"].mean()),
                "executions": executions,
                "adverse_executions": adverse_executions,
                "adverse_event_rate": float(adverse_event_rate),
                "fill_prob_ask_mean": float(g["fill_prob_ask"].mean()),
                "fill_prob_bid_mean": float(g["fill_prob_bid"].mean()),
                "distance_bid_mid_mean": float(g["distance_bid_mid"].mean()),
                "distance_ask_mid_mean": float(g["distance_ask_mid"].mean()),
                "spread_capture_sum": float(g["spread_capture"].sum()),
                "adverse_selection_sum": float(g["adverse_selection"].sum()),
                "inventory_penalty_sum": float(g["inventory_penalty"].sum()),
            }
        )

    return pd.DataFrame(summary_rows).sort_values("pnl_final", ascending=False)


def _calibrate_naive_spread(df):
    """
    Calibra el spread fijo de la estrategia naive.

    La idea es que el benchmark siga siendo simple y no adaptativo,
    pero que no quede artificialmente inactivo en mercados con spreads
    observados extremos.

    Proceso:
    - Se toma la mediana del spread real observado.
    - Si no es valida, se usa 0.02.
    - Se acota entre 0.008 y 0.050.

    Con esto:
    - En mercados liquidos, el naive usa un spread bajo y razonable.
    - En mercados con spread extremo, no se aleja tanto del midprice.
    - Sigue siendo una estrategia fija, no adaptativa.
    """

    market_median_spread = float(df["market_spread"].median())

    if not np.isfinite(market_median_spread) or market_median_spread <= 0:
        market_median_spread = 0.02

    naive_spread = float(np.clip(market_median_spread, 0.008, 0.050))

    return naive_spread, market_median_spread


def _build_path_rows(df, seed):
    """
    Ejecuta el backtest sobre un DataFrame ya preparado y devuelve paths.

    La naive se calibra con la mediana del spread real del mercado analizado,
    pero el spread se acota para que siga siendo un benchmark operativo.
    Esto mantiene la estrategia simple, pero evita que el naive no ejecute
    nunca en mercados con spreads observados extremos.
    """

    naive_spread, market_median_spread = _calibrate_naive_spread(df)

    print("\nSpread real mediano observado:", round(market_median_spread, 6))
    print("Spread fijo usado por la estrategia naive:", round(naive_spread, 6))

    strategies = _new_strategies(naive_spread=naive_spread)
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        for strategy in strategies:
            result = strategy.step(row=row, next_row=next_row, rng=rng)

            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "token_id": row["token_id"],
                    "t": i,

                    # Variables de mercado observadas
                    "best_bid": _safe_get(row, "best_bid"),
                    "best_ask": _safe_get(row, "best_ask"),
                    "midprice": _safe_get(row, "midprice"),
                    "market_spread": _safe_get(row, "market_spread"),
                    "spread_market": _safe_get(row, "market_spread"),
                    "imbalance": _safe_get(row, "imbalance"),
                    "imbalance_change": _safe_get(row, "imbalance_change"),
                    "drift": _safe_get(row, "drift"),
                    "bid_depth": _safe_get(row, "bid_depth"),
                    "ask_depth": _safe_get(row, "ask_depth"),
                    "book_volume_proxy": _safe_get(row, "book_volume_proxy"),
                    "total_depth": _safe_get(row, "total_depth"),

                    # Scores separados
                    "imbalance_change_z": _safe_get(row, "imbalance_change_z"),
                    "drift_z": _safe_get(row, "drift_z"),
                    "imbalance_pressure_score": _safe_get(row, "imbalance_pressure_score"),
                    "drift_pressure_score": _safe_get(row, "drift_pressure_score"),

                    # Senales
                    "flow_z": _safe_get(row, "flow_z"),
                    "flow_z_raw": _safe_get(row, "flow_z_raw"),
                    "flow_z_model": _safe_get(row, "flow_z_model"),
                    "flow_signal_signed": _safe_get(row, "flow_signal_signed"),
                    "vpin": _safe_get(row, "vpin"),
                    "vpin_proxy": _safe_get(row, "vpin_proxy"),
                    "vpin_signal_signed": _safe_get(row, "vpin_signal_signed"),
                    "combined_micro_signal": _safe_get(row, "combined_micro_signal"),
                    "toxic_pressure": _safe_get(row, "toxic_pressure"),
                    "toxicity_score": _safe_get(row, "toxicity_score"),
                    "realized_vol": _safe_get(row, "realized_vol"),

                    # Etiquetas ex post
                    "toxic_flow_label": _safe_get(row, "toxic_flow_label"),
                    "toxic_direction": _safe_get(row, "toxic_direction"),
                    "future_max_mid": _safe_get(row, "future_max_mid"),
                    "future_min_mid": _safe_get(row, "future_min_mid"),

                    # Simulacion
                    "simulation_seed": seed,

                    # Resultado de la estrategia
                    **result,
                }
            )

    return pd.DataFrame(rows)


def run_backtest(
    token_id=None,
    signal_window=SIGNAL_WINDOW,
    save_csv=True,
    seed=123,
    toxic_horizon=TOXIC_HORIZON_TICKS,
    toxic_move_threshold=TOXIC_MOVE_THRESHOLD,
):
    """Ejecuta una simulacion de backtest de las cuatro estrategias."""

    df, token_id = prepare_data(
        token_id=token_id,
        signal_window=signal_window,
        toxic_horizon=toxic_horizon,
        toxic_move_threshold=toxic_move_threshold,
    )

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    paths = _build_path_rows(df=df, seed=seed)
    summary = _summarize_paths(paths)
    descriptive = market_descriptive_summary(df)

    print("\nTABLA 0 - DESCRIPTIVA DEL MERCADO")
    print(descriptive.to_string(index=False))

    print("\nTABLA 1 - RESUMEN DE METRICAS POR ESTRATEGIA")
    print(summary.to_string(index=False))

    if save_csv:
        output_dir = Path(DATA_DIR) / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_token = str(token_id)[:12] if token_id is not None else "market"

        paths_path = output_dir / f"backtest_paths_{safe_token}.csv"
        summary_path = output_dir / f"table1_summary_{safe_token}.csv"
        descriptive_path = output_dir / f"table0_market_descriptive_{safe_token}.csv"

        paths.to_csv(paths_path, index=False)
        summary.to_csv(summary_path, index=False)
        descriptive.to_csv(descriptive_path, index=False)

        print("\nArchivos guardados:")
        print(paths_path)
        print(summary_path)
        print(descriptive_path)

    return paths, summary


def run_backtest_distribution(
    token_id=None,
    signal_window=SIGNAL_WINDOW,
    n_simulations=100,
    save_csv=True,
    toxic_horizon=TOXIC_HORIZON_TICKS,
    toxic_move_threshold=TOXIC_MOVE_THRESHOLD,
):
    """
    Ejecuta muchas simulaciones para obtener distribucion de PnL.

    Misma serie de snapshots reales, distintas semillas aleatorias de fills.
    """

    df, token_id = prepare_data(
        token_id=token_id,
        signal_window=signal_window,
        toxic_horizon=toxic_horizon,
        toxic_move_threshold=toxic_move_threshold,
    )

    if df.empty:
        return pd.DataFrame()

    all_summary = []

    for seed in range(n_simulations):
        paths = _build_path_rows(df=df, seed=seed)
        summary = _summarize_paths(paths)
        summary["simulation_seed"] = seed
        all_summary.append(summary)

    distribution = pd.concat(all_summary, ignore_index=True)

    distribution_summary = (
        distribution
        .groupby("strategy")
        .agg(
            pnl_final_mean=("pnl_final", "mean"),
            pnl_final_std=("pnl_final", "std"),
            pnl_final_p05=("pnl_final", lambda x: x.quantile(0.05)),
            pnl_final_p50=("pnl_final", "median"),
            pnl_final_p95=("pnl_final", lambda x: x.quantile(0.95)),
            sharpe_mean=("sharpe", "mean"),
            max_drawdown_mean=("max_drawdown", "mean"),
            adverse_event_rate_mean=("adverse_event_rate", "mean"),
            executions_mean=("executions", "mean"),
            spread_quoted_mean=("spread_quoted_mean", "mean"),
            signal_signed_abs_mean=("signal_signed_abs_mean", "mean"),
        )
        .reset_index()
        .sort_values("pnl_final_mean", ascending=False)
    )

    print("\nTABLA 2 - DISTRIBUCION DE PNL POR ESTRATEGIA")
    print(distribution_summary.to_string(index=False))

    if save_csv:
        output_dir = Path(DATA_DIR) / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_token = str(token_id)[:12] if token_id is not None else "market"

        distribution_path = output_dir / f"pnl_distribution_{safe_token}.csv"
        distribution_summary_path = output_dir / f"table2_pnl_distribution_summary_{safe_token}.csv"

        distribution.to_csv(distribution_path, index=False)
        distribution_summary.to_csv(distribution_summary_path, index=False)

        print("\nArchivos guardados:")
        print(distribution_path)
        print(distribution_summary_path)

    return distribution