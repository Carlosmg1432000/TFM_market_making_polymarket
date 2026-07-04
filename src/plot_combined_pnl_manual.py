from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("Generando figura conjunta de P&L...")

    # Guardar directamente en el Escritorio
    output_path = Path.home() / "Desktop" / "figure_pnl_final_combined.png"

    estrategias = [
        "Bayes adaptativo",
        "Bayes + VPIN",
        "Heurística de flujo",
        "Naive",
    ]

    # Valores aproximados de tus Figuras 15
    pnl_espana = [8.8, 8.0, 7.8, 5.9]
    pnl_usdc = [8.2, 7.3, 7.8, 6.9]

    x = np.arange(len(estrategias))
    width = 0.35

    plt.figure(figsize=(9, 4.8))

    plt.bar(
        x - width / 2,
        pnl_espana,
        width,
        label="Elecciones anticipadas",
    )

    plt.bar(
        x + width / 2,
        pnl_usdc,
        width,
        label="USDC-USDT",
    )

    plt.ylabel("P&L final")
    plt.xlabel("Estrategia")
    plt.title("Comparación del P&L final por estrategia")

    # Etiquetas menos inclinadas y más limpias
    plt.xticks(x, estrategias, rotation=10, ha="right")

    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Figura generada correctamente.")
    print(f"Ruta exacta: {output_path}")


if __name__ == "__main__":
    main()