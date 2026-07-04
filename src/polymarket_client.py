"""
polymarket_client.py

Cliente para consultar datos públicos de Polymarket.

Incluye:
- funciones para buscar mercados/eventos activos;
- cliente robusto para capturar el order book del CLOB;
- reintentos ante errores SSL o fallos temporales de conexión.

Nota importante:
No todos los token_id que aparecen en la API Gamma tienen necesariamente order
book disponible en el CLOB. Cuando el endpoint /book devuelve 404, se interpreta
como "order book no disponible" y se devuelve None.

Esto es especialmente relevante en la opción 4 del programa, donde se prueban
muchos token_id para recomendar mercados líquidos.
"""

import time
import random
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import SSLError, ConnectionError, Timeout, RequestException


# ============================================================
# URLs BASE
# ============================================================

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"


# ============================================================
# SESIÓN ROBUSTA
# ============================================================

def _create_session() -> requests.Session:
    """
    Crea una sesión HTTP con reintentos para errores temporales.

    Esto ayuda a reducir fallos puntuales de red, errores SSL o respuestas
    temporales del servidor.
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Safari/605.1.15"
            ),
            "Accept": "application/json",
            "Connection": "close",
        }
    )

    return session


# ============================================================
# EVENTOS / MERCADOS ACTIVOS
# ============================================================

def get_active_events(limit: int = 100, offset: int = 0):
    """
    Obtiene eventos activos desde la API Gamma de Polymarket.

    Parameters
    ----------
    limit:
        Número máximo de eventos que se solicitan.
    offset:
        Desplazamiento inicial para paginación simple.

    Returns
    -------
    list
        Lista de eventos activos. Si hay error, devuelve lista vacía.
    """
    session = _create_session()

    url = f"{GAMMA_BASE_URL}/events"
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "offset": offset,
    }

    try:
        response = session.get(url, params=params, timeout=20)
        response.raise_for_status()

        data = response.json()

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ["events", "data", "results"]:
                if key in data and isinstance(data[key], list):
                    return data[key]

        return []

    except RequestException as e:
        print(f"Error obteniendo eventos activos: {e}")
        return []

    except ValueError as e:
        print(f"Error leyendo JSON de eventos activos: {e}")
        return []

    finally:
        session.close()


def search_events(keyword: str, limit: int = 100, offset: int = 0):
    """
    Busca eventos activos por palabra clave.

    Esta función se mantiene porque market_selector.py puede utilizarla.
    """
    events = get_active_events(limit=limit, offset=offset)

    if not keyword:
        return events

    keyword_lower = keyword.lower()
    filtered = []

    for event in events:
        title = str(event.get("title", "")).lower()
        slug = str(event.get("slug", "")).lower()
        description = str(event.get("description", "")).lower()

        if (
            keyword_lower in title
            or keyword_lower in slug
            or keyword_lower in description
        ):
            filtered.append(event)

    return filtered


# ============================================================
# CLIENTE CLOB PARA ORDER BOOK
# ============================================================

class PolymarketClient:
    """
    Cliente para consultar el CLOB de Polymarket.

    Se utiliza una sesión persistente con reintentos para reducir errores
    temporales, especialmente errores SSL como:

    [SSL: UNEXPECTED_EOF_WHILE_READING]
    """

    def __init__(
        self,
        base_url: str = CLOB_BASE_URL,
        timeout: int = 20,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = _create_session()

    def reset_session(self):
        """
        Reinicia la sesión HTTP.

        Esto ayuda cuando el servidor corta conexiones SSL y la sesión queda
        en un estado inestable.
        """
        try:
            self.session.close()
        except Exception:
            pass

        self.session = _create_session()

    def close(self):
        """
        Cierra la sesión HTTP.
        """
        try:
            self.session.close()
        except Exception:
            pass

    def get_order_book(self, token_id: str, silent: bool = False):
        """
        Consulta el order book de un token concreto.

        Parameters
        ----------
        token_id:
            Identificador CLOB del outcome/token.

        silent:
            Si True, no imprime errores esperables. Esto es útil cuando se
            prueban muchos token_id en la opción de recomendar mercados líquidos.

        Returns
        -------
        dict | None
            Devuelve el order book si existe. Devuelve None si no existe,
            si está vacío o si hay un error no recuperable.
        """
        url = f"{self.base_url}/book"
        params = {"token_id": str(token_id)}

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )

                # Caso normal cuando se prueban muchos token_id:
                # el token puede aparecer en Gamma, pero no tener book activo.
                if response.status_code == 404:
                    if not silent:
                        print(f"Order book no disponible para token_id: {token_id}")
                    return None

                response.raise_for_status()
                data = response.json()

                if not isinstance(data, dict):
                    if not silent:
                        print(f"Respuesta inesperada del order book para token_id: {token_id}")
                    return None

                return data

            except (SSLError, ConnectionError, Timeout) as e:
                wait = self.backoff_factor * attempt + random.uniform(0, 1.5)

                if not silent:
                    print(
                        f"Error temporal capturando order book "
                        f"(intento {attempt}/{self.max_retries}). "
                        f"Esperando {wait:.1f}s. Error: {e}"
                    )

                self.reset_session()
                time.sleep(wait)

            except RequestException as e:
                if not silent:
                    print(f"Error HTTP/API capturando order book: {e}")
                return None

            except ValueError as e:
                if not silent:
                    print(f"Error leyendo JSON del order book: {e}")
                return None

        if not silent:
            print("No se pudo capturar el order book tras varios intentos.")

        return None


# ============================================================
# FUNCIÓN DIRECTA POR COMPATIBILIDAD
# ============================================================

def get_order_book(token_id: str, silent: bool = False):
    """
    Función auxiliar por compatibilidad.

    Permite que otros archivos llamen directamente a:

        get_order_book(token_id)

    o a:

        get_order_book(token_id, silent=True)

    sin tener que crear explícitamente una instancia de PolymarketClient.
    """
    client = PolymarketClient()

    try:
        return client.get_order_book(token_id=token_id, silent=silent)
    finally:
        client.close()