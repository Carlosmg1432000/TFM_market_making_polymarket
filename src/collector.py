import sqlite3
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.exceptions import SSLError, ConnectionError, Timeout, RequestException


# ============================================================
# CONFIGURACIÓN
# ============================================================

TOKEN_ID = "91810646921497227084241579668235102205462718984593674693010693975258849328842"

DB_PATH = Path("data") / "market_sim.db"

# Alias necesario porque algunos archivos, como backtester.py,
# pueden importar DEFAULT_DB_PATH desde collector.py.
DEFAULT_DB_PATH = DB_PATH

TABLE_NAME = "orderbook_snapshots"

CAPTURE_INTERVAL_SECONDS = 5

N_SNAPSHOTS = 300

DEPTH_LEVELS = 10

CLOB_BOOK_URL = "https://clob.polymarket.com/book"


# ============================================================
# BASE DE DATOS
# ============================================================

def get_connection():
    """
    Abre la conexión con SQLite y crea la carpeta data si no existe.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def table_exists(cursor, table_name):
    """
    Comprueba si una tabla existe en SQLite.
    """
    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?;
        """,
        (table_name,)
    )
    return cursor.fetchone() is not None


def get_table_columns(cursor, table_name):
    """
    Devuelve las columnas existentes en una tabla.
    """
    cursor.execute(f"PRAGMA table_info({table_name});")
    return [row[1] for row in cursor.fetchall()]


def init_db():
    """
    Crea la tabla de snapshots si no existe.

    Si existe una tabla antigua sin columna id, se renombra como backup
    y se crea una tabla nueva con la estructura correcta.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if table_exists(cursor, TABLE_NAME):
        columns = get_table_columns(cursor, TABLE_NAME)

        if "id" not in columns:
            backup_name = f"{TABLE_NAME}_backup_sin_id_{int(time.time())}"

            print(
                f"Se ha detectado una tabla antigua '{TABLE_NAME}' sin columna id."
            )
            print(f"Renombrando tabla antigua como: {backup_name}")

            cursor.execute(f"ALTER TABLE {TABLE_NAME} RENAME TO {backup_name};")
            conn.commit()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            token_id TEXT NOT NULL,
            best_bid REAL,
            best_ask REAL,
            midprice REAL,
            spread REAL,
            bid_depth REAL,
            ask_depth REAL,
            total_depth REAL,
            imbalance REAL,
            top_bid_size REAL,
            top_ask_size REAL,
            book_volume_proxy REAL
        );
    """)

    conn.commit()
    conn.close()

    print(f"Base de datos inicializada en: {DB_PATH.resolve()}")
    print(f"Tabla usada: {TABLE_NAME}")


def save_snapshot(snapshot):
    """
    Guarda un snapshot en SQLite.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(f"""
        INSERT INTO {TABLE_NAME} (
            timestamp,
            token_id,
            best_bid,
            best_ask,
            midprice,
            spread,
            bid_depth,
            ask_depth,
            total_depth,
            imbalance,
            top_bid_size,
            top_ask_size,
            book_volume_proxy
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        snapshot["timestamp"],
        snapshot["token_id"],
        snapshot["best_bid"],
        snapshot["best_ask"],
        snapshot["midprice"],
        snapshot["spread"],
        snapshot["bid_depth"],
        snapshot["ask_depth"],
        snapshot["total_depth"],
        snapshot["imbalance"],
        snapshot["top_bid_size"],
        snapshot["top_ask_size"],
        snapshot["book_volume_proxy"],
    ))

    conn.commit()
    conn.close()


def count_snapshots():
    """
    Cuenta cuántos snapshots hay guardados en la tabla principal.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
    count = cursor.fetchone()[0]

    conn.close()
    return count


# ============================================================
# CARGA DE DATOS DESDE SQLITE
# ============================================================

def load_data(token_id=None, db_path=DB_PATH):
    """
    Carga los snapshots guardados en SQLite.

    Esta función existe porque backtester.py la importa con:
        from .collector import load_data

    Parámetros
    ----------
    token_id : str | None
        Si se indica, filtra los datos para ese token_id.
        Si es None, carga todos los snapshots disponibles.

    db_path : Path | str
        Ruta de la base de datos SQLite.

    Devuelve
    --------
    pandas.DataFrame
        DataFrame con los snapshots ordenados temporalmente.
    """
    db_path = Path(db_path)

    if not db_path.exists():
        print(f"No existe la base de datos: {db_path.resolve()}")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if not table_exists(cursor, TABLE_NAME):
            print(f"No existe la tabla {TABLE_NAME} en SQLite.")
            conn.close()
            return pd.DataFrame()

        columns = get_table_columns(cursor, TABLE_NAME)

        if "id" not in columns:
            print(
                f"La tabla {TABLE_NAME} existe, pero no tiene columna id. "
                f"Ejecuta collector.py para regenerar la tabla correctamente."
            )
            conn.close()
            return pd.DataFrame()

        if token_id is None:
            query = f"""
                SELECT
                    id,
                    timestamp,
                    token_id,
                    best_bid,
                    best_ask,
                    midprice,
                    spread,
                    bid_depth,
                    ask_depth,
                    total_depth,
                    imbalance,
                    top_bid_size,
                    top_ask_size,
                    book_volume_proxy
                FROM {TABLE_NAME}
                ORDER BY timestamp ASC;
            """
            df = pd.read_sql_query(query, conn)

        else:
            query = f"""
                SELECT
                    id,
                    timestamp,
                    token_id,
                    best_bid,
                    best_ask,
                    midprice,
                    spread,
                    bid_depth,
                    ask_depth,
                    total_depth,
                    imbalance,
                    top_bid_size,
                    top_ask_size,
                    book_volume_proxy
                FROM {TABLE_NAME}
                WHERE token_id = ?
                ORDER BY timestamp ASC;
            """
            df = pd.read_sql_query(query, conn, params=(str(token_id),))

        conn.close()

        if df.empty:
            print("No se han encontrado snapshots para cargar.")
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

        numeric_cols = [
            "best_bid",
            "best_ask",
            "midprice",
            "spread",
            "bid_depth",
            "ask_depth",
            "total_depth",
            "imbalance",
            "top_bid_size",
            "top_ask_size",
            "book_volume_proxy",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["timestamp", "best_bid", "best_ask", "midprice"])

        print(f"Snapshots cargados desde SQLite: {len(df)}")
        print(f"Base de datos usada: {db_path.resolve()}")

        return df

    except Exception as e:
        print(f"Error cargando datos desde SQLite: {e}")
        return pd.DataFrame()


# ============================================================
# CAPTURA DEL ORDER BOOK
# ============================================================

def fetch_order_book(token_id, max_retries=5, base_sleep=2):
    """
    Consulta el libro de órdenes de Polymarket.

    Si hay errores temporales de conexión, reintenta varias veces.
    """
    params = {"token_id": token_id}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                CLOB_BOOK_URL,
                params=params,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json"
                }
            )

            response.raise_for_status()
            return response.json()

        except (SSLError, ConnectionError, Timeout) as e:
            wait = base_sleep * attempt
            print(
                f"Error temporal capturando order book. "
                f"Intento {attempt}/{max_retries}. "
                f"Esperando {wait}s. Error: {e}"
            )
            time.sleep(wait)

        except RequestException as e:
            print(f"Error HTTP/API capturando order book: {e}")
            return None

    print("No se pudo capturar el order book después de varios intentos.")
    return None


# ============================================================
# PROCESADO DEL LIBRO DE ÓRDENES
# ============================================================

def parse_level(level):
    """
    Convierte un nivel del libro en precio y tamaño.

    Polymarket suele devolver niveles como:
        {"price": "...", "size": "..."}

    También se soportan listas o tuplas tipo:
        [price, size]
    """
    if isinstance(level, dict):
        price = float(level.get("price", 0))
        size = float(level.get("size", 0))
        return price, size

    if isinstance(level, (list, tuple)) and len(level) >= 2:
        price = float(level[0])
        size = float(level[1])
        return price, size

    return None, None


def clean_book_side(levels):
    """
    Limpia un lado del libro de órdenes y devuelve:
        [(price, size), ...]
    """
    clean_levels = []

    if not levels:
        return clean_levels

    for level in levels:
        price, size = parse_level(level)

        if price is None or size is None:
            continue

        if price <= 0 or size <= 0:
            continue

        clean_levels.append((price, size))

    return clean_levels


def build_snapshot(book, token_id):
    """
    Construye las variables principales a partir del order book.
    """
    bids = clean_book_side(book.get("bids", []))
    asks = clean_book_side(book.get("asks", []))

    if len(bids) == 0 or len(asks) == 0:
        print("Order book vacío o incompleto. Snapshot omitido.")
        return None

    bids = sorted(bids, key=lambda x: x[0], reverse=True)
    asks = sorted(asks, key=lambda x: x[0])

    best_bid = bids[0][0]
    best_ask = asks[0][0]

    top_bid_size = bids[0][1]
    top_ask_size = asks[0][1]

    midprice = (best_bid + best_ask) / 2
    spread = best_ask - best_bid

    bid_depth = sum(size for _, size in bids[:DEPTH_LEVELS])
    ask_depth = sum(size for _, size in asks[:DEPTH_LEVELS])

    total_depth = bid_depth + ask_depth

    if total_depth > 0:
        imbalance = (bid_depth - ask_depth) / total_depth
    else:
        imbalance = 0.0

    book_volume_proxy = total_depth

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "token_id": str(token_id),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "midprice": midprice,
        "spread": spread,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "total_depth": total_depth,
        "imbalance": imbalance,
        "top_bid_size": top_bid_size,
        "top_ask_size": top_ask_size,
        "book_volume_proxy": book_volume_proxy,
    }

    return snapshot


# ============================================================
# BUCLE PRINCIPAL DE CAPTURA
# ============================================================

def run_collector(
    token_id=TOKEN_ID,
    n_snapshots=N_SNAPSHOTS,
    capture_interval_seconds=CAPTURE_INTERVAL_SECONDS
):
    """
    Ejecuta la captura periódica de snapshots y los guarda en SQLite.
    """
    init_db()

    print("Iniciando captura...")
    print(f"Token_id: {token_id}")
    print(f"Intervalo entre capturas: {capture_interval_seconds}s")
    print(f"Número máximo de snapshots: {n_snapshots}")
    print(f"Niveles usados para profundidad: {DEPTH_LEVELS}")
    print("-" * 80)

    for i in range(n_snapshots):
        book = fetch_order_book(token_id)

        if book is None:
            print(f"{i} Snapshot omitido por error de conexión.")
            time.sleep(capture_interval_seconds)
            continue

        snapshot = build_snapshot(book, token_id)

        if snapshot is None:
            print(f"{i} Snapshot omitido por libro incompleto.")
            time.sleep(capture_interval_seconds)
            continue

        try:
            save_snapshot(snapshot)
            total_saved = count_snapshots()

            print(
                f"{i} guardado | "
                f"mid: {snapshot['midprice']:.4f} | "
                f"spread: {snapshot['spread']:.4f} | "
                f"imbalance: {snapshot['imbalance']:.4f} | "
                f"depth: {snapshot['total_depth']:.2f} | "
                f"total guardados: {total_saved}"
            )

        except Exception as e:
            print(f"{i} Error guardando snapshot en SQLite: {e}")

        time.sleep(capture_interval_seconds)

    print("-" * 80)
    print("Captura finalizada.")
    print(f"Snapshots guardados en total: {count_snapshots()}")
    print(f"Base de datos usada: {DB_PATH.resolve()}")


def collect(
    token_id=TOKEN_ID,
    n_snapshots=N_SNAPSHOTS,
    capture_interval_seconds=CAPTURE_INTERVAL_SECONDS,
    seconds=None,
    n=None,
    **kwargs
):
    """
    Función de compatibilidad con main.py.

    Tu main.py llama:
        collect(token_id=token_id, seconds=seconds, n=n)

    Por eso aquí se aceptan también:
    - seconds -> capture_interval_seconds
    - n -> n_snapshots
    """
    if seconds is not None:
        capture_interval_seconds = seconds

    if n is not None:
        n_snapshots = n

    return run_collector(
        token_id=token_id,
        n_snapshots=n_snapshots,
        capture_interval_seconds=capture_interval_seconds
    )


if __name__ == "__main__":
    run_collector()