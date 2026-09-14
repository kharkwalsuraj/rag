import logging
import ollama

from pathlib import Path
from rag.config import VLM_MODEL_ID, VLM_PROMPT
from rag.main import logger

logger = logging.getLogger(__name__)

def describe_image(img: Path, context: str) -> str:
    if not img.exists():
        raise FileNotFoundError(f"Cannot locate image {str(img.absolute())}")

    prompt = VLM_PROMPT.format(context=context)
    msg = {"role": "user", "content": prompt, "images": [img] }
    logger.info(f"Generating image description for {img}")

    response = ollama.chat(model=VLM_MODEL_ID, messages=[msg])
    logger.debug(f"VLM response: {response}")

    if not response.message.content:
        raise RuntimeError(f"Cannot describe image {img}")

    return response.message.content.strip()
