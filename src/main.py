"""
Punto de entrada principal del proyecto.

Flujo:
1. Buscar mercados por palabra clave.
2. Capturar datos reales de un token_id.
3. Ejecutar backtest y graficas para un token_id.
4. Recomendar mercados liquidos.
5. Ejecutar distribucion de PnL con muchas simulaciones.
"""

from .market_selector import print_candidate_markets, recommend_liquid_markets
from .collector import collect
from .backtester import run_backtest, run_backtest_distribution
from .visualizer import plot_results, plot_pnl_distribution
from .config import SIGNAL_WINDOW, TOXIC_HORIZON_TICKS, TOXIC_MOVE_THRESHOLD


def _optional_int(prompt, default):
    text = input(prompt).strip()

    if text == "":
        return default

    try:
        return int(text)
    except ValueError:
        print(f"Valor no valido. Se usara el valor por defecto: {default}")
        return default


def _optional_float(prompt, default):
    text = input(prompt).strip()

    if text == "":
        return default

    try:
        return float(text)
    except ValueError:
        print(f"Valor no valido. Se usara el valor por defecto: {default}")
        return default


def main():
    """Muestra el menu principal y ejecuta la opcion elegida."""
    print("=" * 90)
    print("TFM POLYMARKET LIVE")
    print("=" * 90)

    print(
        """
1 - Buscar mercados por palabra clave
2 - Capturar datos de UN token_id
3 - Ejecutar backtest y graficas
4 - Recomendar mercados liquidos
5 - Ejecutar distribucion de PnL con muchas simulaciones
"""
    )

    option = input("Elige opcion: ").strip()

    if option == "1":
        keyword = input(
            "Busca mercado, ejemplo btc, eth, sol, bitcoin, ethereum, solana: "
        ).strip()

        if not keyword:
            print("No se ha introducido ninguna palabra clave.")
            return

        print_candidate_markets(keyword=keyword)

    elif option == "2":
        token_id = input("Pega token_id concreto: ").strip()

        if not token_id:
            print("No se ha introducido token_id.")
            return

        seconds = _optional_int(
            "Segundos entre capturas, ENTER para 6: ",
            6,
        )

        n = _optional_int(
            "Numero de capturas, ENTER para 1500: ",
            1500,
        )

        if seconds <= 0:
            print("Los segundos entre capturas deben ser positivos.")
            return

        if n <= 0:
            print("El numero de capturas debe ser positivo.")
            return

        if seconds < 5:
            print(
                "Aviso: se recomienda usar al menos 5-6 segundos entre capturas "
                "para reducir errores de conexion con la API."
            )

        print()
        print("=" * 90)
        print("CONFIGURACION DE CAPTURA")
        print("=" * 90)
        print(f"Token ID: {token_id}")
        print(f"Segundos entre capturas: {seconds}")
        print(f"Numero de capturas: {n}")
        print("=" * 90)
        print()

        try:
            collect(token_id=token_id, seconds=seconds, n=n)
        except KeyboardInterrupt:
            print("\nCaptura interrumpida manualmente por el usuario.")
        except Exception as e:
            print(f"\nError inesperado durante la captura: {e}")

    elif option == "3":
        token_id = input(
            "Pega token_id concreto para backtest, o ENTER para usar el primero guardado: "
        ).strip()

        token_id = token_id if token_id else None

        signal_window = _optional_int(
            f"Ventana de senales, ENTER para {SIGNAL_WINDOW}: ",
            SIGNAL_WINDOW,
        )

        toxic_horizon = _optional_int(
            f"Horizonte toxic flow en snapshots, ENTER para {TOXIC_HORIZON_TICKS}: ",
            TOXIC_HORIZON_TICKS,
        )

        toxic_threshold = _optional_float(
            f"Umbral movimiento toxico, ENTER para {TOXIC_MOVE_THRESHOLD}: ",
            TOXIC_MOVE_THRESHOLD,
        )

        paths, summary = run_backtest(
            token_id=token_id,
            signal_window=signal_window,
            save_csv=True,
            toxic_horizon=toxic_horizon,
            toxic_move_threshold=toxic_threshold,
        )

        if not paths.empty:
            plot_results(paths=paths, summary=summary)
        else:
            print("No se han generado resultados para graficar.")

    elif option == "4":
        recommend_liquid_markets(
            limit_events=100,
            max_pages=2,
            top_n=10,
        )

    elif option == "5":
        token_id = input(
            "Pega token_id concreto para backtest, o ENTER para usar el primero guardado: "
        ).strip()

        token_id = token_id if token_id else None

        signal_window = _optional_int(
            f"Ventana de senales, ENTER para {SIGNAL_WINDOW}: ",
            SIGNAL_WINDOW,
        )

        n_simulations = _optional_int(
            "Numero de simulaciones, ENTER para 100: ",
            100,
        )

        toxic_horizon = _optional_int(
            f"Horizonte toxic flow en snapshots, ENTER para {TOXIC_HORIZON_TICKS}: ",
            TOXIC_HORIZON_TICKS,
        )

        toxic_threshold = _optional_float(
            f"Umbral movimiento toxico, ENTER para {TOXIC_MOVE_THRESHOLD}: ",
            TOXIC_MOVE_THRESHOLD,
        )

        distribution = run_backtest_distribution(
            token_id=token_id,
            signal_window=signal_window,
            n_simulations=n_simulations,
            save_csv=True,
            toxic_horizon=toxic_horizon,
            toxic_move_threshold=toxic_threshold,
        )

        if not distribution.empty:
            plot_pnl_distribution(distribution)
        else:
            print("No se ha generado distribucion de PnL.")

    else:
        print("Opcion no valida.")


if __name__ == "__main__":
    main()