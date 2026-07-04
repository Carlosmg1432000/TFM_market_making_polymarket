"""
Configuracion global del proyecto.

Centraliza rutas, endpoints y parametros generales para evitar valores
hardcodeados en varios modulos.
"""

from pathlib import Path

# Carpeta raiz del proyecto.
ROOT_DIR = Path(__file__).resolve().parents[1]

# Carpeta de datos persistentes.
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Endpoints publicos de Polymarket.
POLYMARKET_EVENTS_URL = "https://gamma-api.polymarket.com/events"
POLYMARKET_BOOK_URL = "https://clob.polymarket.com/book"

# Base SQLite con snapshots del order book.
DATABASE_PATH = str(DATA_DIR / "polymarket_live.db")

# Parametros por defecto de captura.
SNAPSHOT_DELAY_SECONDS = 2
MAX_MARKETS_RETURNED = 10

# Parametros por defecto de senales.
SIGNAL_WINDOW = 50

# Limite solo para la version usada internamente por el modelo.
# El z-score que se grafica y diagnostica sigue siendo flow_z_raw, sin clipping.
FLOW_Z_WINSOR_LIMIT = 5.0

# Horizonte para definir toxic flow ex post.
# Se mide en snapshots/ticks de la muestra, no necesariamente en segundos.
# Se sube de 5 a 10 para no clasificar solo movimientos instantaneos.
TOXIC_HORIZON_TICKS = 10

# Movimiento minimo para considerar desplazamiento adverso relevante.
# En Polymarket los precios estan en [0, 1].
# 0.002 equivale a 0.2 centimos.
TOXIC_MOVE_THRESHOLD = 0.002

# Parametros del modelo de ejecucion.
# La probabilidad de fill cae exponencialmente con la distancia al mid.
EXECUTION_BASE_INTENSITY = 0.65
EXECUTION_DISTANCE_DECAY = 80.0
EXECUTION_PRESSURE_SENSITIVITY = 8.0
EXECUTION_INVENTORY_SENSITIVITY = 0.20