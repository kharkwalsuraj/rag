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
SUPPORTED_EXTENSIONS:set[str] = { ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx" }

VLM_MODEL_ID:str = "qwen2.5vl:3b"
VLM_PROMPT: str = """
You are indexing an image from a document for a RAG system.

Write a standalone, searchable description of the image.

Context from the document surrounding the image:
{context}

Rules:
- Describe what the image conveys and its purpose in the document.
- Use the document context to resolve references such as "this model", "the figure", or "the method".
- For charts or graphs, include the title, axes, labels, units, trends, comparisons, and important values.
- For tables, describe the columns, rows, and important relationships or values.
- For diagrams, describe the components, flow, connections, and relationships.
- For screenshots, describe the application/interface, important elements, and visible text.
- For photos, describe the important objects, people, actions, and visible text.
- Include important text that is clearly readable in the image.
- Do not invent information.
- If something is unreadable or ambiguous, explicitly say so.
- Keep the description concise but sufficiently detailed for semantic search.
- Return only the description. Do not add commentary about these instructions.

Description:
"""
