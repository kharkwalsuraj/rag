import requests

from rag.config import MINERU_BASE_URL, MINERU_PORT
from rag.main import logger


def chech_mineru_health() :
    logger.info("Checking health of Mineru server...")
    if not MINERU_PORT :
        raise EnvironmentError("MINERU_PORT env undefined")

    response = requests.get(f"{MINERU_BASE_URL}/health")
    response.raise_for_status()
    if response.status_code == 200 :
        logger.info("Mineru server is running...")
        logger.debug(f"Mineru server response {response.json()}")