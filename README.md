# TFM - Market Making Adaptativo en Polymarket

Este repositorio contiene el código desarrollado para el Trabajo Fin de Máster **“Algoritmos que reaccionan a ráfagas de liquidez”**.

El objetivo del proyecto es construir una herramienta experimental para analizar estrategias de *market making* adaptativo sobre mercados de predicción de Polymarket. Para ello, se capturan *snapshots* reales del libro de órdenes, se calculan variables de microestructura y se comparan distintas estrategias bajo un entorno común de *backtesting*.

## Descripción general

El proyecto no tiene como finalidad ejecutar órdenes reales en mercado, sino estudiar cómo se comportan diferentes estrategias de cotización cuando operan sobre una misma secuencia histórica de datos.

La herramienta permite:

- Capturar *snapshots* reales del CLOB de Polymarket.
- Almacenar los datos localmente en SQLite.
- Seleccionar mercados con suficiente variabilidad y liquidez para el análisis.
- Calcular variables de mercado como `midprice`, `spread`, profundidad visible, `imbalance` y `drift`.
- Generar señales de presión del flujo.
- Implementar distintas estrategias de *market making*.
- Simular ejecuciones mediante un modelo probabilístico.
- Evaluar resultados mediante métricas de P&L, inventario, probabilidad de ejecución, *spread* cotizado y selección adversa *ex post*.
- Generar tablas resumen y gráficos para analizar el comportamiento de cada estrategia.

## Estrategias implementadas

El sistema compara cuatro estrategias principales:

1. **Naive**
   - Estrategia de referencia.
   - Cotiza de forma simétrica alrededor del `midprice`.
   - Utiliza un *spread* fijo calibrado a partir del mercado observado.

2. **Heurística**
   - Ajusta sus precios de compra y venta a partir de señales directas de presión del flujo.

3. **Bayes adaptativa**
   - Actualiza progresivamente una señal interna de presión.
   - Incorpora cierta memoria sobre el comportamiento reciente del libro de órdenes.

4. **Bayes + VPIN**
   - Estrategia híbrida que combina la señal bayesiana con un proxy inspirado en VPIN.
   - Busca recoger tanto cambios recientes del flujo como desequilibrios más persistentes del CLOB.

## Estructura del proyecto

```text
TFM_market_making_polymarket/
│
├── src/
│   ├── __init__.py
│   ├── backtester.py
│   ├── collector.py
│   ├── config.py
│   ├── main.py
│   ├── market_selector.py
│   ├── plot_combined_pnl_manual.py
│   ├── polymarket_client.py
│   ├── sample_summary.py
│   ├── signals.py
│   ├── strategies.py
│   └── visualizer.py
│
├── data/
│   └── market_data.db
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── results/
│
├── requirements.txt
└── README.md
