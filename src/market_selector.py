"""
market_selector.py

Módulo para buscar y recomendar mercados de Polymarket.

Permite:
1. Buscar mercados por palabra clave y mostrar muchos mercados candidatos.
2. Mostrar token_id y, cuando sea posible, métricas de liquidez/profundidad.
3. Recomendar mercados con order book disponible.
4. Ordenar candidatos por profundidad visible y spread.

Importante:
La profundidad visible NO es volumen negociado real.
Es un proxy de liquidez observable en el order book.
"""

from .polymarket_client import get_active_events, PolymarketClient


# ============================================================
# UTILIDADES
# ============================================================

def _safe_float(x, default=0.0):
    """Convierte a float de forma segura."""
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_str(x):
    """Convierte a string de forma segura."""
    if x is None:
        return ""
    return str(x)


def _get_market_question(event, market):
    """Obtiene un título/pregunta legible del mercado."""
    return (
        market.get("question")
        or market.get("title")
        or event.get("title")
        or event.get("question")
        or "Mercado sin titulo"
    )


def _market_search_text(event, market):
    """
    Construye un texto amplio para buscar por palabra clave.

    Antes puede que solo estuvieras buscando en question/title.
    Ahora se busca también en slug, descripción, outcomes y campos relacionados.
    Esto ayuda a que al buscar btc aparezcan más mercados relacionados.
    """
    parts = [
        event.get("title"),
        event.get("question"),
        event.get("slug"),
        event.get("description"),
        market.get("title"),
        market.get("question"),
        market.get("slug"),
        market.get("description"),
        market.get("outcomePrices"),
        market.get("outcomes"),
        market.get("groupItemTitle"),
        market.get("category"),
    ]

    return " ".join(_safe_str(p).lower() for p in parts)


def _parse_token_ids(raw):
    """
    Convierte distintos formatos de token_id en una lista de strings.

    La API puede devolver token ids como:
    - lista;
    - string separado por comas;
    - string con formato parecido a lista.
    """
    token_ids = []

    if raw is None:
        return token_ids

    if isinstance(raw, list):
        token_ids.extend([str(x) for x in raw if x])
        return token_ids

    if isinstance(raw, str):
        cleaned = (
            raw.replace("[", "")
            .replace("]", "")
            .replace('"', "")
            .replace("'", "")
        )
        token_ids.extend([x.strip() for x in cleaned.split(",") if x.strip()])
        return token_ids

    return token_ids


def _extract_token_ids_from_market(market):
    """
    Extrae token_id candidatos desde un market.

    Se cubren varios formatos porque la respuesta de Gamma puede variar.
    """
    token_ids = []

    for key in ["clobTokenIds", "clob_token_ids"]:
        token_ids.extend(_parse_token_ids(market.get(key)))

    if "tokens" in market and isinstance(market["tokens"], list):
        for token in market["tokens"]:
            if not isinstance(token, dict):
                continue

            token_id = (
                token.get("token_id")
                or token.get("tokenId")
                or token.get("id")
            )

            if token_id:
                token_ids.append(str(token_id))

    if "outcomes" in market and isinstance(market["outcomes"], list):
        for outcome in market["outcomes"]:
            if not isinstance(outcome, dict):
                continue

            token_id = (
                outcome.get("token_id")
                or outcome.get("tokenId")
                or outcome.get("id")
            )

            if token_id:
                token_ids.append(str(token_id))

    seen = set()
    unique = []

    for token_id in token_ids:
        if token_id and token_id not in seen:
            seen.add(token_id)
            unique.append(token_id)

    return unique


def _iter_markets_from_events(events):
    """
    Recorre eventos y devuelve pares (event, market).

    Si un evento no trae lista de markets, se trata el propio evento como market.
    """
    for event in events:
        if not isinstance(event, dict):
            continue

        markets = event.get("markets")

        if isinstance(markets, list) and markets:
            for market in markets:
                if isinstance(market, dict):
                    yield event, market
        else:
            yield event, event


def _load_active_events(limit_events=100, max_pages=1):
    """
    Carga eventos activos usando paginación simple con offset.
    """
    all_events = []

    for page in range(max_pages):
        offset = page * limit_events

        print(f"Cargando eventos pagina {page + 1}/{max_pages}...")

        events = get_active_events(limit=limit_events, offset=offset)

        if not events:
            break

        all_events.extend(events)

        if len(events) < limit_events:
            break

    return all_events


# ============================================================
# CÁLCULO DE MÉTRICAS DE LIQUIDEZ
# ============================================================

def _score_order_book(book):
    """
    Calcula métricas de profundidad visible y liquidez.

    Usa los 5 mejores niveles de cada lado del libro:
    - bid_depth_5niveles
    - ask_depth_5niveles
    - total_depth_5niveles

    No mide volumen real negociado.
    """
    if book is None:
        return None

    bids = book.get("bids", [])
    asks = book.get("asks", [])

    if not bids or not asks:
        return None

    def _price(level):
        if not isinstance(level, dict):
            return 0.0
        return _safe_float(level.get("price"))

    def _size(level):
        if not isinstance(level, dict):
            return 0.0
        return _safe_float(level.get("size"))

    bids_sorted = sorted(bids, key=_price, reverse=True)
    asks_sorted = sorted(asks, key=_price)

    if not bids_sorted or not asks_sorted:
        return None

    best_bid = _price(bids_sorted[0])
    best_ask = _price(asks_sorted[0])

    spread = best_ask - best_bid

    if spread <= 0:
        return None

    midprice = (best_bid + best_ask) / 2.0

    top_bid_size = _size(bids_sorted[0])
    top_ask_size = _size(asks_sorted[0])
    top_depth = top_bid_size + top_ask_size

    bid_depth_5 = sum(_size(level) for level in bids_sorted[:5])
    ask_depth_5 = sum(_size(level) for level in asks_sorted[:5])
    total_depth_5 = bid_depth_5 + ask_depth_5

    if total_depth_5 <= 0:
        return None

    imbalance_5 = (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5 + 1e-8)

    bid_levels = len(bids_sorted)
    ask_levels = len(asks_sorted)

    liquidity_score = total_depth_5 / max(spread, 1e-6)
    top_liquidity_score = top_depth / max(spread, 1e-6)

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "midprice": midprice,
        "spread": spread,
        "top_bid_size": top_bid_size,
        "top_ask_size": top_ask_size,
        "top_depth": top_depth,
        "bid_depth_5niveles": bid_depth_5,
        "ask_depth_5niveles": ask_depth_5,
        "total_depth_5niveles": total_depth_5,
        "bid_levels_visibles": bid_levels,
        "ask_levels_visibles": ask_levels,
        "imbalance_5niveles": imbalance_5,
        "liquidity_score": liquidity_score,
        "top_liquidity_score": top_liquidity_score,
        "score": liquidity_score,
    }


def _print_market_metrics_compact(token_id, metrics):
    """
    Imprime métricas compactas para la opción 1.
    """
    print("  token_id:", token_id)

    if metrics is None:
        print("  estado: sin order book disponible o no comprobado")
        return

    print("  PRECIOS")
    print("    best_bid:", round(metrics["best_bid"], 6))
    print("    best_ask:", round(metrics["best_ask"], 6))
    print("    midprice:", round(metrics["midprice"], 6))
    print("    spread:", round(metrics["spread"], 6))

    print("  LIQUIDEZ / PROFUNDIDAD VISIBLE")
    print("    top_depth:", round(metrics["top_depth"], 2))
    print("    bid_depth_5niveles:", round(metrics["bid_depth_5niveles"], 2))
    print("    ask_depth_5niveles:", round(metrics["ask_depth_5niveles"], 2))
    print("    total_depth_5niveles:", round(metrics["total_depth_5niveles"], 2))
    print("    bid_levels_visibles:", metrics["bid_levels_visibles"])
    print("    ask_levels_visibles:", metrics["ask_levels_visibles"])

    print("  SEÑALES BÁSICAS")
    print("    imbalance_5niveles:", round(metrics["imbalance_5niveles"], 4))
    print("    liquidity_score:", round(metrics["liquidity_score"], 2))


def _print_market_metrics_full(index, candidate, title="CANDIDATO VÁLIDO"):
    """
    Imprime las métricas completas de un candidato.
    """
    print("\n" + "=" * 110)
    print(f"{title} #{index}")
    print("=" * 110)

    print("Mercado:", candidate["question"])
    print("token_id:", candidate["token_id"])

    print("\nPRECIOS")
    print("best_bid:", round(candidate["best_bid"], 6))
    print("best_ask:", round(candidate["best_ask"], 6))
    print("midprice:", round(candidate["midprice"], 6))
    print("spread:", round(candidate["spread"], 6))

    print("\nLIQUIDEZ / PROFUNDIDAD VISIBLE")
    print("top_bid_size:", round(candidate["top_bid_size"], 2))
    print("top_ask_size:", round(candidate["top_ask_size"], 2))
    print("top_depth:", round(candidate["top_depth"], 2))
    print("bid_depth_5niveles:", round(candidate["bid_depth_5niveles"], 2))
    print("ask_depth_5niveles:", round(candidate["ask_depth_5niveles"], 2))
    print("total_depth_5niveles:", round(candidate["total_depth_5niveles"], 2))
    print("bid_levels_visibles:", candidate["bid_levels_visibles"])
    print("ask_levels_visibles:", candidate["ask_levels_visibles"])

    print("\nSEÑALES BÁSICAS")
    print("imbalance_5niveles:", round(candidate["imbalance_5niveles"], 4))
    print("liquidity_score:", round(candidate["liquidity_score"], 2))
    print("top_liquidity_score:", round(candidate["top_liquidity_score"], 2))


# ============================================================
# OPCIÓN 1 - BUSCAR MERCADOS POR PALABRA CLAVE
# ============================================================

def print_candidate_markets(
    keyword="",
    limit_events=200,
    max_pages=5,
    max_markets_to_print=60,
    check_order_books=True,
    max_tokens_to_check=120,
):
    """
    Busca mercados por palabra clave y muestra token_id candidatos.

    Esta versión vuelve a mostrar muchos más mercados, como antes, porque:
    - busca en varias páginas;
    - aumenta el límite de eventos;
    - busca en más campos del evento/mercado;
    - no se queda solo con los 3 primeros.

    Además, cuando puede, muestra liquidez/profundidad visible.
    """
    keyword = (keyword or "").lower().strip()

    print("\nBuscando mercados...")
    print("Palabra clave:", keyword if keyword else "(sin filtro)")
    print("Eventos por página:", limit_events)
    print("Páginas máximas:", max_pages)
    print("Máximo de mercados a mostrar:", max_markets_to_print)

    if check_order_books:
        print("\nTambién se intentará mostrar liquidez/profundidad visible si el book está disponible.")
        print("Nota: no es volumen real negociado; es profundidad visible del order book.\n")

    events = _load_active_events(
        limit_events=limit_events,
        max_pages=max_pages,
    )

    print("\nEventos cargados:", len(events))

    if not events:
        print("No se encontraron eventos activos.")
        return

    client = PolymarketClient(
        timeout=5,
        max_retries=1,
        backoff_factor=0.3,
    )

    rows_printed = 0
    checked_tokens = 0
    valid_books = 0
    unavailable_books = 0
    total_matching_markets = 0

    try:
        for event, market in _iter_markets_from_events(events):
            search_text = _market_search_text(event, market)

            if keyword and keyword not in search_text:
                continue

            token_ids = _extract_token_ids_from_market(market)

            if not token_ids:
                continue

            total_matching_markets += 1

            if rows_printed >= max_markets_to_print:
                continue

            rows_printed += 1

            question = _get_market_question(event, market)

            print("\n" + "=" * 110)
            print(f"MERCADO #{rows_printed}")
            print("=" * 110)
            print("Mercado:", question)
            print("Evento:", event.get("title", ""))
            print("Slug evento:", event.get("slug", ""))
            print("Token IDs candidatos:")

            for token_id in token_ids:
                if not check_order_books:
                    print("-", token_id)
                    continue

                if checked_tokens >= max_tokens_to_check:
                    print("  token_id:", token_id)
                    print("  estado: no comprobado, límite de checks de liquidez alcanzado")
                    continue

                checked_tokens += 1

                book = client.get_order_book(token_id, silent=True)
                metrics = _score_order_book(book)

                if metrics is None:
                    unavailable_books += 1
                else:
                    valid_books += 1

                _print_market_metrics_compact(token_id, metrics)
                print("")

    finally:
        client.close()

    if total_matching_markets == 0:
        print("No se encontraron mercados que coincidan con la búsqueda.")
        return

    print("\n" + "=" * 110)
    print("RESUMEN OPCIÓN 1")
    print("=" * 110)
    print("Mercados coincidentes encontrados:", total_matching_markets)
    print("Mercados impresos:", rows_printed)

    if rows_printed < total_matching_markets:
        print("Nota: había más mercados, pero se alcanzó el límite de impresión.")

    if check_order_books:
        print("Token_id comprobados con order book:", checked_tokens)
        print("Token_id con book válido:", valid_books)
        print("Token_id sin book disponible o vacío:", unavailable_books)

    print("\nCÓMO USAR ESTO")
    print("1. Escoge mercados con spread bajo.")
    print("2. Prioriza total_depth_5niveles alto.")
    print("3. Mira que top_depth no sea demasiado bajo.")
    print("4. Usa imbalance para ver si el libro está muy cargado hacia bid o ask.")
    print("5. Copia el token_id elegido y úsalo en la opción 2 para capturar datos.")

    print("\nNota para el TFM:")
    print("Estas métricas son profundidad visible / proxy de liquidez.")
    print("No son volumen negociado real.")


# ============================================================
# OPCIÓN 4 - RECOMENDAR MERCADOS LÍQUIDOS
# ============================================================

def recommend_liquid_markets(
    limit_events=100,
    max_pages=3,
    top_n=10,
    max_tokens_to_check=250,
    min_valid_candidates=10,
):
    """
    Recomienda mercados con order book disponible.

    Esta versión imprime la liquidez/profundidad visible en cuanto encuentra
    un candidato válido.
    """
    print("\nBuscando mercados liquidos...")
    print("Se mostrarán las métricas de liquidez en cuanto se encuentre un candidato válido.")
    print("Pulsa Ctrl+C si quieres cancelar.\n")

    events = _load_active_events(
        limit_events=limit_events,
        max_pages=max_pages,
    )

    if not events:
        print("No se encontraron eventos activos.")
        return []

    print("Eventos cargados:", len(events))

    candidates = []
    checked_tokens = 0
    unavailable_books = 0
    invalid_books = 0

    client = PolymarketClient(
        timeout=5,
        max_retries=1,
        backoff_factor=0.3,
    )

    try:
        for event, market in _iter_markets_from_events(events):
            question = _get_market_question(event, market)
            token_ids = _extract_token_ids_from_market(market)

            if not token_ids:
                continue

            for token_id in token_ids:
                if checked_tokens >= max_tokens_to_check:
                    print("\nLímite de tokens revisados alcanzado.")
                    break

                checked_tokens += 1

                if checked_tokens % 10 == 0:
                    print(
                        f"Progreso: {checked_tokens} tokens revisados | "
                        f"{len(candidates)} candidatos válidos | "
                        f"{unavailable_books} sin book"
                    )

                book = client.get_order_book(token_id, silent=True)

                if book is None:
                    unavailable_books += 1
                    continue

                metrics = _score_order_book(book)

                if metrics is None:
                    invalid_books += 1
                    continue

                candidate = {
                    "question": question,
                    "token_id": token_id,
                    **metrics,
                }

                candidates.append(candidate)

                _print_market_metrics_full(
                    index=len(candidates),
                    candidate=candidate,
                    title="CANDIDATO VÁLIDO ENCONTRADO",
                )

                if len(candidates) >= min_valid_candidates:
                    print("\nYa se han encontrado suficientes candidatos válidos.")
                    break

            if checked_tokens >= max_tokens_to_check:
                break

            if len(candidates) >= min_valid_candidates:
                break

    finally:
        client.close()

    print("\n" + "=" * 110)
    print("RESUMEN DE BÚSQUEDA")
    print("=" * 110)
    print("Tokens revisados:", checked_tokens)
    print("Candidatos válidos:", len(candidates))
    print("Tokens sin order book:", unavailable_books)
    print("Books inválidos o sin bids/asks:", invalid_books)

    if not candidates:
        print("\nNo se encontraron mercados con order book disponible en esta búsqueda.")
        print("\nPosibles causas:")
        print("- Muchos token_id de Gamma no tienen book activo en CLOB.")
        print("- La búsqueda revisó pocos eventos.")
        print("- Puede que los primeros eventos devueltos por la API no sean buenos candidatos.")
        print("\nPrueba aumentando limit_events o buscando manualmente con la opción 1.")
        return []

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    top_candidates = candidates[:top_n]

    print("\n" + "=" * 110)
    print("RANKING FINAL DE MERCADOS RECOMENDADOS")
    print("=" * 110)

    for i, row in enumerate(top_candidates, start=1):
        print(
            f"{i}. mid={row['midprice']:.4f} | "
            f"spread={row['spread']:.4f} | "
            f"depth_5niv={row['total_depth_5niveles']:.2f} | "
            f"imbalance={row['imbalance_5niveles']:.3f} | "
            f"score={row['liquidity_score']:.2f} | "
            f"token={row['token_id'][:16]}..."
        )
        print("   Mercado:", row["question"])
        print("   token_id:", row["token_id"])

    print("\n" + "=" * 110)
    print("CÓMO INTERPRETAR LAS MÉTRICAS")
    print("=" * 110)
    print("spread bajo                 -> más fácil cotizar cerca del mercado.")
    print("total_depth_5niveles alto    -> más profundidad visible cerca del precio.")
    print("top_depth alto               -> más tamaño justo en el mejor bid/ask.")
    print("imbalance positivo           -> más profundidad en el lado comprador.")
    print("imbalance negativo           -> más profundidad en el lado vendedor.")
    print("liquidity_score alto         -> mucha profundidad relativa al spread.")

    print("\nNota para el TFM:")
    print("Esto no es volumen negociado real.")
    print("Es profundidad visible del order book y se usa como proxy de liquidez/actividad.")

    return top_candidates