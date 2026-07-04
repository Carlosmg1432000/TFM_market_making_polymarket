"""
Estrategias de market making.

Logica:
- Cada estrategia genera cotizaciones bid y ask.
- La estrategia naive mantiene un spread fijo, pero no ejecuta siempre.
- La ejecucion depende de una probabilidad de cierre.
- La probabilidad de ejecucion cae con la distancia al midprice.

Cambio importante:
- La estrategia naive ya no usa un spread fijo escrito a mano.
- El spread fijo de la naive se recibe desde el backtester.
- El backtester lo calcula como la mediana del spread real observado en el mercado.
"""

from math import lgamma, log, exp
import numpy as np

from .config import (
    EXECUTION_BASE_INTENSITY,
    EXECUTION_DISTANCE_DECAY,
    EXECUTION_PRESSURE_SENSITIVITY,
    EXECUTION_INVENTORY_SENSITIVITY,
)


def poisson_logpmf(k, lam):
    """Log-probabilidad de observar k bajo una Poisson(lambda)."""
    lam = max(lam, 1e-12)
    return -lam + k * log(lam) - lgamma(k + 1)


class BayesFilter:
    """
    Filtro bayesiano para estimar probabilidad de regimen toxico.

    Bayes estima la magnitud del riesgo en [0, 1].
    La direccion se toma de la senal firmada disponible.
    """

    def __init__(self, p_cc=0.990, p_tt=0.970, lam_c=0.35, lam_t=1.25, pi0=0.05):
        self.p_cc = p_cc
        self.p_tt = p_tt
        self.lam_c = lam_c
        self.lam_t = lam_t
        self.pi = pi0

    def update(self, shock_magnitude):
        """Actualiza pi = P(regimen toxico | shock observado)."""

        p_ct = 1.0 - self.p_cc
        pi_pred = (1.0 - self.pi) * p_ct + self.pi * self.p_tt

        k = int(max(0, min(8, round(float(shock_magnitude)))))

        like_t = exp(poisson_logpmf(k, self.lam_t))
        like_c = exp(poisson_logpmf(k, self.lam_c))

        num = like_t * pi_pred
        den = num + like_c * (1.0 - pi_pred)

        self.pi = num / den if den > 0 else pi_pred
        self.pi = max(0.0, min(1.0, self.pi))

        return self.pi


class BaseStrategy:
    """Clase base con logica comun de market making."""

    def __init__(
        self,
        name,
        spread_c=0.025,
        spread_t=0.050,
        inventory_skew_coef=0.00030,
        inventory_penalty_lambda=0.00002,
        max_inventory=60,
        execution_base_intensity=EXECUTION_BASE_INTENSITY,
        execution_distance_decay=EXECUTION_DISTANCE_DECAY,
    ):
        self.name = name
        self.spread_c = spread_c
        self.spread_t = spread_t
        self.inventory_skew_coef = inventory_skew_coef
        self.inventory_penalty_lambda = inventory_penalty_lambda
        self.max_inventory = max_inventory
        self.execution_base_intensity = execution_base_intensity
        self.execution_distance_decay = execution_distance_decay

        self.cash = 0.0
        self.inventory = 0.0

    def signal(self, row):
        """
        Devuelve una senal firmada en [-1, 1].

        Positivo:
            presion compradora / riesgo de subida.

        Negativo:
            presion vendedora / riesgo de bajada.
        """
        return 0.0

    def quote(self, midprice, signal_signed):
        """
        Genera bid y ask teoricos.

        La magnitud de la senal abre el spread.
        El signo de la senal desplaza el centro de cotizacion.
        """

        signal_signed = max(-1.0, min(1.0, float(signal_signed)))
        defense_score = abs(signal_signed)

        # Spread base entre regimen tranquilo y defensivo.
        spread = self.spread_c + (self.spread_t - self.spread_c) * defense_score

        # Ajuste adicional por inventario.
        inventory_widening = 0.00008 * abs(self.inventory)
        final_spread = spread + inventory_widening

        # Si tenemos inventario positivo, bajamos ligeramente el centro para favorecer ventas.
        inventory_skew = self.inventory_skew_coef * self.inventory

        # Si la senal es positiva, hay riesgo de subida: se desplaza el centro hacia arriba.
        # Si es negativa, hay riesgo de bajada: se desplaza hacia abajo.
        toxicity_skew = 0.25 * final_spread * signal_signed

        reservation_price = midprice - inventory_skew + toxicity_skew

        bid = reservation_price - final_spread / 2.0
        ask = reservation_price + final_spread / 2.0

        bid = max(0.0, min(1.0, bid))
        ask = max(0.0, min(1.0, ask))

        if bid > ask:
            bid, ask = ask, bid

        return bid, ask, final_spread

    def mtm(self, midprice):
        """Mark-to-market con penalizacion cuadratica de inventario."""

        inventory_penalty = self.inventory_penalty_lambda * (self.inventory ** 2)
        return self.cash + self.inventory * midprice - inventory_penalty

    def fill_probability(self, side, quote_price, midprice, pressure, signal_signed):
        """
        Probabilidad simulada de ejecucion.

        Cuanto mas lejos esta la cotizacion del midprice, menor sera la probabilidad.
        Esto evita que una estrategia gane mecanicamente por abrir siempre mas spread.
        """

        signal_signed = max(-1.0, min(1.0, float(signal_signed)))
        defense_score = abs(signal_signed)

        distance_to_mid = abs(float(quote_price) - float(midprice))

        # Decaimiento exponencial con la distancia al mid.
        distance_component = np.exp(-self.execution_distance_decay * distance_to_mid)

        # Cuanto mas defensiva es la estrategia, menor agresividad de ejecucion.
        defense_component = 1.0 - 0.50 * defense_score

        # Presion compradora aumenta la probabilidad de que nos levanten el ask.
        # Presion vendedora aumenta la probabilidad de que nos golpeen el bid.
        if side == "ask":
            pressure_component = 1.0 + EXECUTION_PRESSURE_SENSITIVITY * max(0.0, pressure)

            # Si tenemos inventario positivo, vender ayuda a reducirlo.
            inventory_component = 1.0 + EXECUTION_INVENTORY_SENSITIVITY * max(
                0.0,
                np.tanh(self.inventory / 20.0),
            )

        else:
            pressure_component = 1.0 + EXECUTION_PRESSURE_SENSITIVITY * max(0.0, -pressure)

            # Si tenemos inventario negativo, comprar ayuda a reducirlo.
            inventory_component = 1.0 + EXECUTION_INVENTORY_SENSITIVITY * max(
                0.0,
                -np.tanh(self.inventory / 20.0),
            )

        prob = (
            self.execution_base_intensity
            * distance_component
            * defense_component
            * pressure_component
            * inventory_component
        )

        return float(max(0.0, min(0.95, prob)))

    def step(self, row, next_row, rng):
        """Ejecuta un paso del backtest."""

        mid = float(row["midprice"])
        next_mid = float(next_row["midprice"])

        imbalance = float(row.get("imbalance", 0.0))
        next_imbalance = float(next_row.get("imbalance", 0.0))

        drift = float(row.get("drift", 0.0))
        signal_signed = self.signal(row)
        defense_score = abs(signal_signed)

        bid, ask, quoted_spread = self.quote(mid, signal_signed)

        # Presion observable para el modelo de ejecucion.
        # Se mantiene sencilla: movimiento reciente + desequilibrio del libro.
        market_move = next_mid - mid
        pressure = market_move + 0.002 * imbalance + 0.001 * next_imbalance + drift

        executed_buy = 0
        executed_sell = 0

        allow_buy_inventory = self.inventory < self.max_inventory
        allow_sell_inventory = self.inventory > -self.max_inventory

        fill_prob_ask = self.fill_probability(
            side="ask",
            quote_price=ask,
            midprice=mid,
            pressure=pressure,
            signal_signed=signal_signed,
        )

        fill_prob_bid = self.fill_probability(
            side="bid",
            quote_price=bid,
            midprice=mid,
            pressure=pressure,
            signal_signed=signal_signed,
        )

        # Cliente compra contra nuestro ask: nosotros vendemos.
        if allow_sell_inventory and rng.random() < fill_prob_ask:
            self.inventory -= 1.0
            self.cash += ask
            executed_buy = 1

        # Cliente vende contra nuestro bid: nosotros compramos.
        if allow_buy_inventory and rng.random() < fill_prob_bid:
            self.inventory += 1.0
            self.cash -= bid
            executed_sell = 1

        toxic_direction = int(row.get("toxic_direction", 0))
        toxic_label = int(row.get("toxic_flow_label", 0))

        adverse_event = 0

        # Vendimos en ask y despues hubo movimiento adverso al alza.
        if executed_buy and toxic_label and toxic_direction > 0:
            adverse_event = 1

        # Compramos en bid y despues hubo movimiento adverso a la baja.
        if executed_sell and toxic_label and toxic_direction < 0:
            adverse_event = 1

        spread_capture = (executed_buy + executed_sell) * quoted_spread / 2.0

        future_max_mid = float(row.get("future_max_mid", next_mid))
        future_min_mid = float(row.get("future_min_mid", next_mid))

        adverse_selection = 0.0

        if executed_buy:
            adverse_selection -= max(0.0, future_max_mid - ask)

        if executed_sell:
            adverse_selection -= max(0.0, bid - future_min_mid)

        inventory_penalty = self.inventory_penalty_lambda * (self.inventory ** 2)

        pnl = self.mtm(next_mid)

        return {
            "strategy": self.name,
            "midprice": mid,
            "next_midprice": next_mid,
            "signal_signed": signal_signed,
            "defense_score": defense_score,
            "directional_pressure": signal_signed,
            "signal": defense_score,
            "bid": bid,
            "ask": ask,
            "spread": quoted_spread,
            "distance_bid_mid": abs(bid - mid),
            "distance_ask_mid": abs(ask - mid),
            "inventory": self.inventory,
            "cash": self.cash,
            "pnl": pnl,
            "executed_buy": executed_buy,
            "executed_sell": executed_sell,
            "fill_prob_ask": fill_prob_ask,
            "fill_prob_bid": fill_prob_bid,
            "adverse_event": adverse_event,
            "spread_capture": spread_capture,
            "adverse_selection": adverse_selection,
            "inventory_penalty": inventory_penalty,
            "pressure": pressure,
        }


class NaiveStrategy(BaseStrategy):
    """
    Benchmark naive.

    Mantiene un spread fijo calibrado a partir del spread real observado
    en el mercado analizado.

    No compra ni vende siempre: solo cotiza y luego se aplica la probabilidad
    de ejecucion comun al resto de estrategias.
    """

    def __init__(self, fixed_spread=0.02):
        super().__init__(
            name="naive",
            spread_c=fixed_spread,
            spread_t=fixed_spread,
        )

    def signal(self, row):
        return 0.0


class HeuristicStrategy(BaseStrategy):
    """
    Estrategia heuristica.

    Usa una combinacion simetrica entre:
    - senal de flujo
    - proxy VPIN firmado

    No se usan ponderaciones arbitrarias tipo 0.7 / 0.3.
    """

    def __init__(self):
        super().__init__(
            name="heuristic_flow",
            spread_c=0.025,
            spread_t=0.050,
        )

    def signal(self, row):
        flow_signed = float(row.get("flow_signal_signed", 0.0))
        vpin_signed = float(row.get("vpin_signal_signed", 0.0))

        signal_signed = np.mean([flow_signed, vpin_signed])

        return max(-1.0, min(1.0, signal_signed))


class BayesStrategy(BaseStrategy):
    """
    Estrategia Bayes adaptativa.

    Usa un filtro bayesiano para estimar la magnitud del riesgo.
    La direccion se obtiene de la senal de flujo firmada.
    """

    def __init__(self):
        super().__init__(
            name="adaptive_bayes",
            spread_c=0.025,
            spread_t=0.050,
        )
        self.filter = BayesFilter()

    def signal(self, row):
        flow_signed = float(row.get("flow_signal_signed", 0.0))
        shock = abs(float(row.get("flow_z_model", 0.0)))

        pi = self.filter.update(shock)

        signal_signed = pi * np.sign(flow_signed)

        return max(-1.0, min(1.0, signal_signed))


class BayesVPINStrategy(BaseStrategy):
    """
    Estrategia Bayes + VPIN proxy.

    Combina de forma simetrica:
    - probabilidad bayesiana firmada
    - VPIN proxy firmado
    """

    def __init__(self):
        super().__init__(
            name="adaptive_bayes_vpin",
            spread_c=0.025,
            spread_t=0.050,
        )
        self.filter = BayesFilter()

    def signal(self, row):
        flow_signed = float(row.get("flow_signal_signed", 0.0))
        vpin_signed = float(row.get("vpin_signal_signed", 0.0))
        shock = abs(float(row.get("flow_z_model", 0.0)))

        pi = self.filter.update(shock)

        bayes_signed = pi * np.sign(flow_signed)

        signal_signed = np.mean([bayes_signed, vpin_signed])

        return max(-1.0, min(1.0, signal_signed))