"""
Calculo de senales de microestructura.

Version revisada:
- Se separan dos indicadores principales: imbalance y drift.
- No se usan ponderaciones arbitrarias tipo 0.7 / 0.3.
- Todas las senales principales se expresan de forma firmada.
- Valor positivo: presion compradora / riesgo de subida.
- Valor negativo: presion vendedora / riesgo de bajada.
- La magnitud indica intensidad de la senal.
"""

import numpy as np
import pandas as pd

from .config import FLOW_Z_WINSOR_LIMIT, TOXIC_HORIZON_TICKS, TOXIC_MOVE_THRESHOLD


def _safe_numeric(series, default=0.0):
    """Convierte una serie a numerica y rellena NaN."""
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _rolling_z(series, window=50, min_periods=10, min_std=1e-4):
    """
    Calcula un z-score movil evitando explosiones cuando la desviacion
    tipica local es practicamente cero.

    Esto es importante en mercados con muchos snapshots planos, ya que si
    la desviacion movil es casi nula, cualquier pequeno cambio puede generar
    un z-score artificialmente extremo.
    """
    series = pd.to_numeric(series, errors="coerce").fillna(0.0)

    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()

    z = (series - mean) / std

    z = z.where(std >= min_std, 0.0)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return z


def add_signals(
    df,
    window=50,
    z_winsor_limit=FLOW_Z_WINSOR_LIMIT,
    toxic_horizon=TOXIC_HORIZON_TICKS,
    toxic_move_threshold=TOXIC_MOVE_THRESHOLD,
):
    """
    Anade senales al DataFrame.

    Senales principales:
    - imbalance_change_z: z-score del cambio del imbalance.
    - drift_z: z-score del cambio del midprice.
    - imbalance_pressure_score: senal firmada basada en imbalance.
    - drift_pressure_score: senal firmada basada en drift.
    - flow_z_raw: z-score estadistico sin clipping de la presion combinada.
    - flow_z_model: version recortada solo para alimentar el modelo.
    - flow_signal_signed: senal de flujo firmada en [-1, 1].
    - vpin_signal_signed: proxy VPIN firmado en [-1, 1].
    - combined_micro_signal: senal combinada sin ponderaciones arbitrarias.
    """

    d = df.copy().reset_index(drop=True)

    # Asegurar columnas numericas basicas.
    for col in [
        "midprice",
        "best_bid",
        "best_ask",
        "spread",
        "imbalance",
        "bid_depth",
        "ask_depth",
    ]:
        if col not in d.columns:
            d[col] = 0.0
        d[col] = _safe_numeric(d[col])

    # Spread real observado en el mercado.
    d["market_spread"] = d["best_ask"] - d["best_bid"]

    # Profundidad visible del libro.
    if "total_depth" not in d.columns:
        d["total_depth"] = d["bid_depth"] + d["ask_depth"]
    else:
        d["total_depth"] = _safe_numeric(d["total_depth"])
        d["total_depth"] = d["total_depth"].where(
            d["total_depth"].notna() & (d["total_depth"] > 0),
            d["bid_depth"] + d["ask_depth"],
        )

    if "book_volume_proxy" not in d.columns:
        d["book_volume_proxy"] = d["total_depth"]
    else:
        d["book_volume_proxy"] = _safe_numeric(d["book_volume_proxy"])
        d["book_volume_proxy"] = d["book_volume_proxy"].where(
            d["book_volume_proxy"].notna() & (d["book_volume_proxy"] > 0),
            d["total_depth"],
        )

    # ------------------------------------------------------------------
    # 1. VARIABLES DESCRIPTIVAS DEL MERCADO
    # ------------------------------------------------------------------

    # Drift del midprice entre snapshots.
    d["mid_return"] = d["midprice"].diff().fillna(0.0)
    d["drift"] = d["mid_return"]

    # Cambio del spread real.
    d["spread_change"] = d["market_spread"].diff().fillna(0.0)

    # Cambio del imbalance.
    d["imbalance_change"] = d["imbalance"].diff().fillna(0.0)

    # ------------------------------------------------------------------
    # 2. SCORES SEPARADOS: IMBALANCE Y DRIFT
    # ------------------------------------------------------------------

    # Z-score del cambio de imbalance.
    d["imbalance_change_z"] = _rolling_z(
        d["imbalance_change"],
        window=window,
        min_periods=10,
    )

    # Z-score del drift.
    d["drift_z"] = _rolling_z(
        d["drift"],
        window=window,
        min_periods=10,
    )

    # Limitamos scores extremos para evitar que una observacion domine todo el modelo.
    d["imbalance_change_z_model"] = d["imbalance_change_z"].clip(
        -z_winsor_limit,
        z_winsor_limit,
    )

    d["drift_z_model"] = d["drift_z"].clip(
        -z_winsor_limit,
        z_winsor_limit,
    )

    # Scores firmados en [-1, 1].
    d["imbalance_pressure_score"] = np.tanh(d["imbalance_change_z_model"] / 3.0)
    d["drift_pressure_score"] = np.tanh(d["drift_z_model"] / 3.0)

    # ------------------------------------------------------------------
    # 3. SENAL DE FLUJO SIN PONDERACIONES ARBITRARIAS
    # ------------------------------------------------------------------

    # Combinacion simetrica de imbalance y drift.
    # Se divide por sqrt(2) para mantener una escala razonable.
    d["signed_flow_proxy"] = (
        d["imbalance_change_z"] + d["drift_z"]
    ) / np.sqrt(2.0)

    # Z-score diagnostico de la presion combinada.
    d["flow_z_raw"] = _rolling_z(
        d["signed_flow_proxy"],
        window=window,
        min_periods=10,
    )

    # Este z-score se grafica y diagnostica, no se recorta.
    d["flow_z"] = d["flow_z_raw"]

    # Version recortada solo para alimentar el modelo.
    d["flow_z_model"] = d["flow_z_raw"].clip(-z_winsor_limit, z_winsor_limit)

    # Senal firmada comparable.
    d["flow_signal_signed"] = np.tanh(d["flow_z_model"] / 3.0)

    # ------------------------------------------------------------------
    # 4. VPIN PROXY FIRMADO
    # ------------------------------------------------------------------

    # VPIN academico requeriria trades clasificados.
    # Aqui se usa un proxy basado en desequilibrio persistente del libro.
    d["vpin_proxy"] = (
        d["imbalance"]
        .abs()
        .rolling(window=window, min_periods=1)
        .mean()
        .clip(0.0, 1.0)
    )

    d["vpin_signal_signed"] = (
        d["imbalance"]
        .rolling(window=window, min_periods=1)
        .mean()
        .fillna(0.0)
        .clip(-1.0, 1.0)
    )

    # Alias para compatibilidad con codigo anterior.
    d["vpin"] = d["vpin_proxy"]

    # ------------------------------------------------------------------
    # 5. SENAL MICRO COMBINADA SIN 0.7 / 0.3
    # ------------------------------------------------------------------

    # Combinamos de forma simetrica las senales disponibles.
    # No se asignan pesos calibrados manualmente.
    signal_components = pd.concat(
        [
            d["flow_signal_signed"],
            d["vpin_signal_signed"],
        ],
        axis=1,
    )

    d["combined_micro_signal"] = (
        signal_components
        .mean(axis=1)
        .fillna(0.0)
        .clip(-1.0, 1.0)
    )

    d["toxicity_score"] = d["combined_micro_signal"].abs()
    d["toxic_pressure"] = d["combined_micro_signal"]

    # Volatilidad realizada del midprice.
    d["realized_vol"] = (
        d["mid_return"]
        .rolling(window=window, min_periods=1)
        .std()
        .fillna(0.0)
    )

    # Etiquetas ex post de flujo potencialmente toxico.
    d = add_toxic_flow_labels(
        d,
        horizon=toxic_horizon,
        move_threshold=toxic_move_threshold,
    )

    return d


def add_toxic_flow_labels(df, horizon=10, move_threshold=0.002):
    """
    Define flujo potencialmente toxico ex post mirando una ventana futura.

    toxic_up:
        el precio futuro maximo supera el ask actual por al menos threshold.

    toxic_down:
        el precio futuro minimo cae por debajo del bid actual por al menos threshold.

    Importante:
    Estas etiquetas no se usan para que la estrategia mire el futuro.
    Se emplean despues para evaluar la calidad de las ejecuciones simuladas.
    """

    d = df.copy()
    n = len(d)

    future_max_mid = np.full(n, np.nan)
    future_min_mid = np.full(n, np.nan)

    mids = d["midprice"].to_numpy(dtype=float)

    for i in range(n):
        start = i + 1
        end = min(n, i + horizon + 1)

        if start >= end:
            continue

        future_window = mids[start:end]
        future_max_mid[i] = np.max(future_window)
        future_min_mid[i] = np.min(future_window)

    d["future_max_mid"] = future_max_mid
    d["future_min_mid"] = future_min_mid

    d["toxic_up"] = (d["future_max_mid"] - d["best_ask"]) > move_threshold
    d["toxic_down"] = (d["best_bid"] - d["future_min_mid"]) > move_threshold

    d["toxic_flow_label"] = (d["toxic_up"] | d["toxic_down"]).astype(int)

    d["toxic_direction"] = 0
    d.loc[d["toxic_up"] & ~d["toxic_down"], "toxic_direction"] = 1
    d.loc[d["toxic_down"] & ~d["toxic_up"], "toxic_direction"] = -1

    both = d["toxic_up"] & d["toxic_down"]

    up_move = d["future_max_mid"] - d["midprice"]
    down_move = d["midprice"] - d["future_min_mid"]

    d.loc[both & (up_move >= down_move), "toxic_direction"] = 1
    d.loc[both & (down_move > up_move), "toxic_direction"] = -1

    return d


def zscore_diagnostics(df):
    """Devuelve diagnosticos del flow_z_raw."""

    if "flow_z_raw" not in df.columns:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "share_abs_gt_3": 0.0,
        }

    z = df["flow_z_raw"].replace([np.inf, -np.inf], np.nan).dropna()

    if z.empty:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "share_abs_gt_3": 0.0,
        }

    return {
        "mean": float(z.mean()),
        "std": float(z.std()),
        "p01": float(z.quantile(0.01)),
        "p05": float(z.quantile(0.05)),
        "p50": float(z.quantile(0.50)),
        "p95": float(z.quantile(0.95)),
        "p99": float(z.quantile(0.99)),
        "share_abs_gt_3": float((z.abs() > 3.0).mean()),
    }