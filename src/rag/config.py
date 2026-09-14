import os
from dotenv import load_dotenv

_ = load_dotenv()

MINERU_PORT:str=os.environ["MINERU_PORT"]
MINERU_BASE_URL:str=f"http://localhost:{MINERU_PORT}"
MINERU_PAYLOAD:dict[str, str] = {
    "image_analysis": "true",
    "client_side_output_generation": "false",
    "return_middle_json": "false",
    "return_model_output": "false",
    "return_md": "true",
    "return_images": "true",
    "end_page_id": "99999",
    "effort": "medium",
    "parse_method": "auto",
    "start_page_id": "0",
    "lang_list": "en",
    "server_url": "string",
    "return_content_list": "false",
    "backend": "hybrid-engine",
    "table_enable": "true",
    "response_format_zip": "false",
    "return_original_file": "false",
    "formula_enable": "true",
}
SUPPORTED_EXTENSIONS:set[str] = { ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".png", ".jpg", ".jpeg", ".webp" }

VLM_MODEL_ID:str = "qwen2.5vl:3b"
VLM_PROMPT:str = """
You are indexing images for a RAG system. Write a standalone, searchable description of this image.

Context from the document around the image:
{context}

Rules:
- Describe what the image CONVEYS, not just what it looks like.
- If chart/graph/table: include title, axes, labels, units, trends, key values.
- If diagram: describe the flow, components, and relationships.
- If photo/screenshot: describe main objects, people, actions, and visible text.
- Use the context to resolve references like "this model", "the figure".
- Do NOT hallucinate. If something is unreadable, say so.
- Output type markdown.

Description:
"""
