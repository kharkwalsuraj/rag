import ollama

from pathlib import Path
from rag.config import VLM_MODEL_ID, VLM_PROMPT


def describe_image(img: Path, context: str) -> str:
    prompt = VLM_PROMPT.format(context=context)
    msg = {"role": "user", "content": prompt, "images": [img]}
    response = ollama.chat(model=VLM_MODEL_ID, messages=[msg])
    if not response.message.content :
        raise RuntimeError(f"Can not describe image {img}")
    return response.message.content.strip()

if __name__ == "__main__":
    ...
