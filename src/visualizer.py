"""
Visualizacion de resultados.

Orden revisado:
1. Primero se muestran los datos observados del mercado.
2. Despues se muestran las senales.
3. Luego se muestran cotizaciones, ejecucion, inventario, seleccion adversa y PnL.

Esto sigue la logica:
datos -> senales -> ejecucion -> estrategias -> resultados.
"""

import matplotlib.pyplot as plt


STRATEGY_LABELS = {
    "naive": "Naive",
    "heuristic_flow": "Heuristica de flujo",
    "adaptive_bayes": "Bayes adaptativo",
    "adaptive_bayes_vpin": "Bayes + VPIN adaptativo",
}


def _pretty_strategy_name(strategy):
    """Devuelve un nombre legible para las estrategias."""

    return STRATEGY_LABELS.get(strategy, strategy)


def _first_strategy_view(paths):
    """
    Devuelve una sola trayectoria para graficar variables de mercado.
    Como las variables de mercado son iguales para todas las estrategias,
    basta con tomar una de ellas.
    """

    preferred = paths[paths["strategy"] == "adaptive_bayes_vpin"].copy()

    if not preferred.empty:
        return preferred

    first_strategy = paths["strategy"].iloc[0]
    return paths[paths["strategy"] == first_strategy].copy()


def plot_results(paths, summary=None):
    """Genera las figuras principales del experimento."""

    if paths.empty:
        print("No hay resultados para graficar.")
        return

    token_id = str(paths["token_id"].iloc[0])
    title_suffix = f" | token: {token_id[:10]}..."

    one = _first_strategy_view(paths)

    # -------------------------------------------------------------
    # FIGURA 1: MIDPRICE OBSERVADO
    # -------------------------------------------------------------
    if not one.empty:
        plt.figure(figsize=(12, 5))

        plt.plot(
            one["t"],
            one["midprice"],
            label="Midprice observado",
        )

        plt.title("Figura 1 - Evolucion del midprice observado" + title_suffix)
        plt.xlabel("t, indice de snapshot")
        plt.ylabel("Midprice")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------
    # FIGURA 2: SPREAD REAL DEL MERCADO
    # -------------------------------------------------------------
    if not one.empty and "market_spread" in one.columns:
        plt.figure(figsize=(12, 5))

        plt.plot(
            one["t"],
            one["market_spread"],
            label="Spread real del mercado",
        )

        plt.title("Figura 2 - Spread real observado en el order book" + title_suffix)
        plt.xlabel("t, indice de snapshot")
        plt.ylabel("Best ask - best bid")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------
    # FIGURA 3: PROFUNDIDAD VISIBLE
    # -------------------------------------------------------------
    if not one.empty and "total_depth" in one.columns:
        plt.figure(figsize=(12, 5))

        plt.plot(
            one["t"],
            one["total_depth"],
            label="Profundidad visible total",
        )

        plt.title("Figura 3 - Profundidad visible del order book" + title_suffix)
        plt.xlabel("t, indice de snapshot")
        plt.ylabel("Profundidad visible")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------
    # FIGURA 4: IMBALANCE OBSERVADO
    # -------------------------------------------------------------
    if not one.empty and "imbalance" in one.columns:
        plt.figure(figsize=(12, 5))

        plt.plot(
            one["t"],
            one["imbalance"],
            label="Imbalance del libro",
        )

        plt.axhline(0.0, linewidth=1)

        plt.title("Figura 4 - Imbalance observado en el order book" + title_suffix)
        plt.xlabel("t, indice de snapshot")
        plt.ylabel("Imbalance")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------
    # FIGURA 5: DRIFT DEL MIDPRICE
    # -------------------------------------------------------------
    if not one.empty and "drift" in one.columns:
        plt.figure(figsize=(12, 5))

        plt.plot(
            one["t"],
            one["drift"],
            label="Drift del midprice",
        )

        plt.axhline(0.0, linewidth=1)

        plt.title("Figura 5 - Drift del midprice entre snapshots" + title_suffix)
        plt.xlabel("t, indice de snapshot")
        plt.ylabel("Variacion del midprice")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------
    # FIGURA 6: SCORES SEPARADOS DE PRESION
    # -------------------------------------------------------------
    if not one.empty:
        plt.figure(figsize=(12, 5))

        if "imbalance_pressure_score" in one.columns:
            plt.plot(
                one["t"],
                one["imbalance_pressure_score"],
                label="Score de imbalance",
            )

        if "drift_pressure_score" in one.columns:
            plt.plot(
                one["t"],
                one["drift_pressure_score"],
                label="Score de drift",
            )

        plt.axhline(0.0, linewidth=1)
        plt.axhline(1.0, linewidth=0.5)
        plt.axhline(-1.0, linewidth=0.5)

        plt.title("Figura 6 - Scores separados de imbalance y drift" + title_suffix)
        plt.xlabel("t, indice de snapshot")
        plt.ylabel("Score en [-1, 1]")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------
    # FIGURA 7: SENALES FIRMADAS Y FLUJO TOXICO EX POST
    # -------------------------------------------------------------
    if not one.empty:
        plt.figure(figsize=(12, 5))

        plt.plot(
            one["t"],
            one["flow_signal_signed"],
            label="Senal de flujo firmada",
        )

        plt.plot(
            one["t"],
            one["vpin_signal_signed"],
            label="Proxy VPIN firmado",
        )

        plt.plot(
            one["t"],
            one["combined_micro_signal"],
            label="Senal combinada",
        )

        toxic_points = one[one["toxic_flow_label"] == 1]

        if not toxic_points.empty:
            plt.scatter(
                toxic_points["t"],
                toxic_points["toxic_direction"],
                label="Direccion del flujo toxico ex post",
                s=15,
            )

        plt.axhline(0.0, linewidth=1)
        plt.axhline(1.0, linewidth=0.5)
        plt.axhline(-1.0, linewidth=0.5)

        plt.title("Figura 7 - Senales firmadas y flujo toxico ex post" + title_suffix)
        plt.xlabel("t, indice de snapshot")
        plt.ylabel("Senal en [-1, 1]")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------
    # FIGURA 8: DIAGNOSTICO DEL Z-SCORE RAW
    # -------------------------------------------------------------
    if not one.empty:
        plt.figure(figsize=(12, 5))

        plt.plot(
            one["t"],
            one["flow_z_raw"],
            label="Z-score de flujo sin recorte",
        )

        plt.axhline(0.0, linewidth=1)
        plt.axhline(3.0, linewidth=0.5)
        plt.axhline(-3.0, linewidth=0.5)

        plt.title("Figura 8 - Diagnostico del z-score sin clipping" + title_suffix)
        plt.xlabel("t, indice de snapshot")
        plt.ylabel("Z-score")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------
    # FIGURA 9: SPREAD COTIZADO POR ESTRATEGIA
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 5))

    for strategy, g in paths.groupby("strategy"):
        plt.plot(
            g["t"],
            g["spread"],
            label=_pretty_strategy_name(strategy),
        )

    plt.title("Figura 9 - Spread cotizado por estrategia" + title_suffix)
    plt.xlabel("t, indice de snapshot")
    plt.ylabel("Spread cotizado")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # FIGURA 10: SPREAD REAL VS SPREAD COTIZADO MEDIO
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 5))

    if "market_spread" in one.columns:
        plt.plot(
            one["t"],
            one["market_spread"],
            label="Spread real del mercado",
        )

    quoted_mean = (
        paths
        .groupby("t")["spread"]
        .mean()
        .reset_index()
    )

    plt.plot(
        quoted_mean["t"],
        quoted_mean["spread"],
        label="Spread cotizado medio de estrategias",
    )

    plt.title("Figura 10 - Spread real frente a spread cotizado medio" + title_suffix)
    plt.xlabel("t, indice de snapshot")
    plt.ylabel("Spread")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # FIGURA 11: PROBABILIDAD MEDIA DE EJECUCION
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 5))

    for strategy, g in paths.groupby("strategy"):
        fill_prob = (g["fill_prob_ask"] + g["fill_prob_bid"]) / 2.0

        plt.plot(
            g["t"],
            fill_prob,
            label=_pretty_strategy_name(strategy),
        )

    plt.title("Figura 11 - Probabilidad media de ejecucion" + title_suffix)
    plt.xlabel("t, indice de snapshot")
    plt.ylabel("Probabilidad")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # FIGURA 12: INVENTARIO
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 5))

    for strategy, g in paths.groupby("strategy"):
        plt.plot(
            g["t"],
            g["inventory"],
            label=_pretty_strategy_name(strategy),
        )

    plt.title("Figura 12 - Inventario por estrategia" + title_suffix)
    plt.xlabel("t, indice de snapshot")
    plt.ylabel("Inventario")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # FIGURA 13: SELECCION ADVERSA EX POST
    # -------------------------------------------------------------
    if summary is not None and not summary.empty:
        adverse = summary[["strategy", "adverse_event_rate"]].copy()
    else:
        executions = (
            paths.groupby("strategy")[["executed_buy", "executed_sell"]]
            .sum()
            .sum(axis=1)
        )

        adverse_counts = paths.groupby("strategy")["adverse_event"].sum()
        adverse = (adverse_counts / executions.replace(0, 1)).reset_index()
        adverse.columns = ["strategy", "adverse_event_rate"]

    adverse["strategy_label"] = adverse["strategy"].apply(_pretty_strategy_name)

    plt.figure(figsize=(10, 5))

    plt.bar(
        adverse["strategy_label"],
        adverse["adverse_event_rate"],
    )

    plt.title("Figura 13 - Tasa de seleccion adversa ex post")
    plt.xlabel("Estrategia")
    plt.ylabel("Tasa de eventos adversos")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # FIGURA 14: PNL ACUMULADO
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 5))

    for strategy, g in paths.groupby("strategy"):
        plt.plot(
            g["t"],
            g["pnl"],
            label=_pretty_strategy_name(strategy),
        )

    plt.title("Figura 14 - PnL acumulado por estrategia" + title_suffix)
    plt.xlabel("t, indice de snapshot")
    plt.ylabel("PnL")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # FIGURA 15: PNL FINAL
    # -------------------------------------------------------------
    if summary is not None and not summary.empty:
        pnl_final = summary[["strategy", "pnl_final"]].copy()
    else:
        pnl_final = (
            paths.groupby("strategy")["pnl"]
            .last()
            .reset_index()
            .rename(columns={"pnl": "pnl_final"})
        )

    pnl_final["strategy_label"] = pnl_final["strategy"].apply(_pretty_strategy_name)

    plt.figure(figsize=(10, 5))

    plt.bar(
        pnl_final["strategy_label"],
        pnl_final["pnl_final"],
    )

    plt.title("Figura 15 - PnL final por estrategia")
    plt.xlabel("Estrategia")
    plt.ylabel("PnL final")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def plot_pnl_distribution(distribution):
    """Grafica la distribucion de PnL final por estrategia."""

    if distribution.empty:
        print("No hay distribucion para graficar.")
        return

    ordered = []
    labels = []

    for strategy, g in distribution.groupby("strategy"):
        ordered.append(g["pnl_final"].values)
        labels.append(_pretty_strategy_name(strategy))

    plt.figure(figsize=(11, 5))

    plt.boxplot(
        ordered,
        labels=labels,
    )

    plt.title("Distribucion de PnL final por estrategia")
    plt.xlabel("Estrategia")
    plt.ylabel("PnL final")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()