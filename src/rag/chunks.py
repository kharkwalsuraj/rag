"""Upload local PDFs to MinerU, download full results, and build KB chunks."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from collections.abc import Iterable, Sequence
from dotenv import load_dotenv

load_dotenv()
API_BASE = "https://mineru.net/api/v4"
MAX_FILE_BYTES = 200_000_000
MAX_BATCH_FILES = 50
MAX_EXTRACTED_BYTES = 10 * 1024 * 1024 * 1024
TRANSIENT_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
TERMINAL_STATES = {"done", "failed"}
USER_AGENT = "mineru-kb-packager/1.0"
PROGRESS_HEARTBEAT_SECONDS = 60.0
RESULT_ZIP_RETRY_DELAYS = (2, 5, 10)
RESULT_ZIP_ATTEMPTS = len(RESULT_ZIP_RETRY_DELAYS) + 1


class MinerUError(RuntimeError):
    """A user-facing MinerU acquisition error."""


class InvalidResultZipError(MinerUError):
    """The result URL returned a body that is not a valid Zip archive."""


@dataclass(frozen=True)
class Job:
    source_pdf: Path
    data_id: str
    result_dir: Path
    project_root: Path


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(min(2 ** attempt, 8))


def _decode_json(raw: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MinerUError(f"{context}returned an unrecognized response") from exc
    if not isinstance(value, dict):
        raise MinerUError(f"{context}returned an invalid format")
    return value


def api_request(
    method: str,
    path: str,
    token: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: int = 60,
    retries: int = 4,
) -> dict[str, Any]:
    """Call the MinerU JSON API without ever logging the bearer token."""
    body = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    }
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    url = f"{API_BASE}{path}"
    last_error: BaseException | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = _decode_json(response.read(), "MinerU API")
        except urllib.error.HTTPError as exc:
            last_error = exc
            raw = exc.read()
            if exc.code in TRANSIENT_HTTP_STATUS and attempt + 1 < retries:
                _sleep_before_retry(attempt)
                continue
            detail = ""
            try:
                error_value = _decode_json(raw, "MinerU API")
                detail = str(error_value.get("msg") or "")
            except MinerUError:
                pass
            suffix = f": {detail}" if detail else ""
            raise MinerUError(f"MinerU API HTTP {exc.code}{suffix}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                _sleep_before_retry(attempt)
                continue
            raise MinerUError("Unable to connect to the MinerU API; check your network and retry") from exc

        if result.get("code") not in (0, "0"):
            code = result.get("code", "unknown")
            message = result.get("msg") or "Unknown error"
            trace_id = result.get("trace_id")
            trace = f"，trace_id={trace_id}" if trace_id else ""
            raise MinerUError(f"MinerU API error {code}: {message}{trace}")
        data = result.get("data")
        if not isinstance(data, dict):
            raise MinerUError("MinerU API response is missing the data object")
        return data

    raise MinerUError("MinerU API request failed") from last_error


def upload_file(upload_url: str, source: Path, *, retries: int = 4) -> None:
    """Stream a file to a signed URL, deliberately omitting Content-Type."""
    parsed = urllib.parse.urlsplit(upload_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise MinerUError(f"{source.name}: MinerU returned an invalid upload URL")
    target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))

    for attempt in range(retries):
        connection_class = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_class(parsed.hostname, parsed.port, timeout=120)
        try:
            connection.putrequest("PUT", target, skip_accept_encoding=True)
            connection.putheader("Content-Length", str(source.stat().st_size))
            connection.putheader("User-Agent", USER_AGENT)
            connection.endheaders()
            with source.open("rb") as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    connection.send(chunk)
            response = connection.getresponse()
            response.read()
            if 200 <= response.status < 300:
                return
            if response.status in TRANSIENT_HTTP_STATUS and attempt + 1 < retries:
                _sleep_before_retry(attempt)
                continue
            raise MinerUError(f"{source.name}: upload failed, HTTP {response.status}")
        except (TimeoutError, OSError, http.client.HTTPException) as exc:
            if attempt + 1 < retries:
                _sleep_before_retry(attempt)
                continue
            raise MinerUError(f"{source.name}: connection interrupted during upload") from exc
        finally:
            connection.close()


def download_file(url: str, destination: Path, *, retries: int = 4) -> None:
    """Download a result Zip without exposing its signed URL."""
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with destination.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            if destination.exists():
                destination.unlink()
            status = getattr(exc, "code", None)
            transient = status is None or status in TRANSIENT_HTTP_STATUS
            if transient and attempt + 1 < retries:
                _sleep_before_retry(attempt)
                continue
            suffix = f"，HTTP {status}" if status else ""
            raise MinerUError(f"failed to download MinerU result{suffix}") from exc


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract a Zip after rejecting traversal, links, and unreasonable expansion."""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            total_size = 0
            for info in archive.infolist():
                member = PurePosixPath(info.filename.replace("\\", "/"))
                if member.is_absolute() or ".." in member.parts or "\x00" in info.filename:
                    raise MinerUError("MinerU result Zip contains an unsafe path")
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise MinerUError("MinerU result Zip contains a disallowed symbolic link")
                total_size += info.file_size
                if total_size > MAX_EXTRACTED_BYTES:
                    raise MinerUError("MinerU result is unusually large after extraction; extraction stopped")
            archive.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise InvalidResultZipError("MinerU returned a result that is not a valid Zip") from exc


def find_result_root(extracted_dir: Path) -> Path:
    candidates = set()
    for pattern in ("content_list_v2.json", "*_content_list_v2.json"):
        for path in extracted_dir.rglob(pattern):
            if path.is_file():
                candidates.add(path.parent)
    if not candidates:
        raise MinerUError("complete result is missing content_list_v2.json")
    if len(candidates) != 1:
        names = ", ".join(sorted(str(path.relative_to(extracted_dir)) for path in candidates))
        raise MinerUError(f"multiple structured document directories found in complete result: {names}")
    return candidates.pop()


def validate_result_dir(result_dir: Path) -> None:
    candidates = list(result_dir.glob("*_content_list_v2.json"))
    exact = result_dir / "content_list_v2.json"
    if exact.is_file():
        candidates.append(exact)
    if len(candidates) != 1:
        raise MinerUError(f"{result_dir}: did not find exactly one content_list_v2.json")
    try:
        with candidates[0].open("r", encoding="utf-8") as stream:
            content = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MinerUError(f"{result_dir}: content_list_v2.json could not be read") from exc
    if not isinstance(content, list):
        raise MinerUError(f"{result_dir}: content_list_v2.json top level is not a page array")


def install_result(
    job: Job,
    result: dict[str, Any],
    batch_id: str,
    settings: dict[str, Any],
) -> None:
    zip_url = result.get("full_zip_url")
    if not isinstance(zip_url, str) or not zip_url:
        raise MinerUError(f"{job.source_pdf.name}: completed state is missing full_zip_url")

    job.result_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mineru-download-", dir=job.result_dir.parent) as tmp:
        temporary = Path(tmp)
        for attempt in range(RESULT_ZIP_ATTEMPTS):
            zip_path = temporary / f"result-{attempt + 1}.zip"
            extracted_dir = temporary / f"extracted-{attempt + 1}"
            extracted_dir.mkdir()

            print(
                f"[{job.source_pdf.name}] downloading complete result "
                f"({attempt + 1}/{RESULT_ZIP_ATTEMPTS})"
            )
            download_file(zip_url, zip_path)
            try:
                safe_extract_zip(zip_path, extracted_dir)
            except InvalidResultZipError as exc:
                if attempt + 1 >= RESULT_ZIP_ATTEMPTS:
                    raise MinerUError(
                        f"MinerU result was not a valid Zip after {RESULT_ZIP_ATTEMPTS} consecutive attempts"
                    ) from exc
                delay = RESULT_ZIP_RETRY_DELAYS[attempt]
                print(
                    f"[{job.source_pdf.name}] result Zip is not available yet,"
                    f"{delay} seconds before retrying the download"
                )
                time.sleep(delay)
                continue

            result_root = find_result_root(extracted_dir)
            validate_result_dir(result_root)

            if job.result_dir.exists():
                raise MinerUError(f"{job.result_dir}: target directory already exists; refusing to overwrite")
            result_root.rename(job.result_dir)
            break

    manifest = {
        "source_pdf": str(job.source_pdf),
        "data_id": job.data_id,
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "mineru_file_name": result.get("file_name") or job.source_pdf.name,
    }
    manifest_path = job.result_dir / "mineru_acquisition.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def make_data_id(path: Path) -> str:
    file_stat = path.stat()
    identity = f"{path}:{file_stat.st_size}:{file_stat.st_mtime_ns}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem).strip("-.") or "document"
    return f"{slug[:110]}-{digest}"[:128]


def expand_pdf_inputs(values: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in values:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            paths.extend(
                sorted(
                    item.resolve()
                    for item in path.iterdir()
                    if item.is_file() and item.suffix.lower() == ".pdf"
                )
            )
        elif path.is_file():
            paths.append(path)
        else:
            raise MinerUError(f"input does not exist: {path}")

    unique: list[Path] = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if path.suffix.lower() != ".pdf":
            raise MinerUError(f"only PDF input is supported: {path}")
        size = path.stat().st_size
        if size == 0:
            raise MinerUError(f"PDF is empty: {path}")
        if size > MAX_FILE_BYTES:
            raise MinerUError(f"PDF exceeds MinerU's 200 MB limit: {path}")
        unique.append(path)
    if not unique:
        raise MinerUError("no PDF files found")
    return unique


def build_jobs(pdf_paths: Sequence[Path], output_root: Path | None) -> list[Job]:
    names = [path.name for path in pdf_paths]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise MinerUError(f"duplicate PDF names found; cannot map batch results: {', '.join(duplicate_names)}")

    jobs = []
    for source in pdf_paths:
        project_root = output_root if output_root else source.parent
        result_dir = project_root / f"{source.name}-mineru"
        jobs.append(
            Job(
                source_pdf=source,
                data_id=make_data_id(source),
                result_dir=result_dir,
                project_root=project_root,
            )
        )
    return jobs


def discover_existing_jobs(output_root: Path | None) -> list[Job]:
    """Build convert-only jobs by scanning an existing result root."""
    if output_root is None:
        raise MinerUError("when PDF input is omitted, --convert-only must also specify --output-root")
    if not output_root.is_dir():
        raise MinerUError(f"MinerU result root does not exist: {output_root}")

    result_dirs = sorted(
        path for path in output_root.iterdir()
        if path.is_dir() and path.name.endswith("-mineru")
    )
    if not result_dirs:
        raise MinerUError(f"{output_root}: no *-mineru result directories found")

    jobs = []
    for result_dir in result_dirs:
        source_name = result_dir.name[:-len("-mineru")]
        jobs.append(
            Job(
                source_pdf=output_root / source_name,
                data_id="",
                result_dir=result_dir,
                project_root=output_root,
            )
        )
    return jobs


def chunks(values: Sequence[Job], size: int) -> Iterable[Sequence[Job]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def create_batch(
    jobs: Sequence[Job],
    token: str,
    settings: dict[str, Any],
) -> tuple[str, list[str]]:
    files = []
    for job in jobs:
        item: dict[str, Any] = {"name": job.source_pdf.name, "data_id": job.data_id}
        if settings["is_ocr"]:
            item["is_ocr"] = True
        files.append(item)
    payload = {
        "files": files,
        "model_version": settings["model_version"],
        "enable_formula": settings["enable_formula"],
        "enable_table": settings["enable_table"],
    }
    if settings.get("language"):
        payload["language"] = settings["language"]
    data = api_request("POST", "/file-urls/batch", token, payload)
    batch_id = data.get("batch_id")
    urls = data.get("file_urls")
    if not isinstance(batch_id, str) or not batch_id:
        raise MinerUError("upload URL request succeeded, but response is missing batch_id")
    if not isinstance(urls, list) or len(urls) != len(jobs):
        raise MinerUError("number of upload URLs returned by MinerU does not match the number of PDFs")
    if not all(isinstance(url, str) and url for url in urls):
        raise MinerUError("MinerU returned an invalid upload URL")
    return batch_id, urls


def poll_batch(
    batch_id: str,
    jobs: Sequence[Job],
    token: str,
    *,
    interval: float,
    timeout: float,
) -> dict[str, dict[str, Any]]:
    started_at = time.monotonic()
    deadline = started_at + timeout
    previous_progress: dict[str, tuple[str, Any, Any]] = {}
    last_reported_at: dict[str, float] = {}
    expected = {job.data_id: job for job in jobs}

    while time.monotonic() < deadline:
        data = api_request("GET", f"/extract-results/batch/{batch_id}", token)
        raw_results = data.get("extract_result") or []
        if not isinstance(raw_results, list):
            raise MinerUError("extract_result in batch query response has an invalid format")

        by_data_id: dict[str, dict[str, Any]] = {}
        by_name: dict[str, dict[str, Any]] = {}
        for value in raw_results:
            if not isinstance(value, dict):
                continue
            if value.get("data_id"):
                by_data_id[str(value["data_id"])] = value
            if value.get("file_name"):
                by_name[str(value["file_name"])] = value

        resolved: dict[str, dict[str, Any]] = {}
        all_terminal = True
        now = time.monotonic()
        for data_id, job in expected.items():
            value = by_data_id.get(data_id) or by_name.get(job.source_pdf.name)
            state = str(value.get("state") or "waiting-file") if value else "waiting-file"
            current = None
            total = None
            if value and isinstance(value.get("extract_progress"), dict):
                current = value["extract_progress"].get("extracted_pages")
                total = value["extract_progress"].get("total_pages")

            signature = (state, current, total)
            changed = previous_progress.get(data_id) != signature
            heartbeat_due = (
                now - last_reported_at.get(data_id, started_at)
                >= PROGRESS_HEARTBEAT_SECONDS
            )
            if changed or heartbeat_due:
                progress = (
                    f" ({current}/{total} pages)"
                    if current is not None and total is not None
                    else ""
                )
                heartbeat = ""
                if heartbeat_due and not changed and state not in TERMINAL_STATES:
                    heartbeat = f"，waited {int(now - started_at)} seconds"
                print(f"[{job.source_pdf.name}] {state}{progress}{heartbeat}")
                previous_progress[data_id] = signature
                last_reported_at[data_id] = now
            if value:
                resolved[data_id] = value
            if state not in TERMINAL_STATES:
                all_terminal = False

        if all_terminal and len(resolved) == len(jobs):
            return resolved
        time.sleep(interval)

    raise MinerUError(f"batch {batch_id} did not complete within {int(timeout)} seconds")


def run_converter(job: Job, shared_output: Path | None) -> None:
    converter = Path(__file__).with_name("converter.py")
    command = [sys.executable, "-u", str(converter), str(job.result_dir)]
    if shared_output:
        command.extend(["--shared-output", str(shared_output)])
    print(f"[{job.source_pdf.name}] building knowledge-base chunks")
    try:
        subprocess.run(command, cwd=job.project_root, check=True)
    except subprocess.CalledProcessError as exc:
        raise MinerUError(f"{job.source_pdf.name}: converter.py execution failed") from exc
    validate_kb_output(job)


def validate_kb_output(job: Job) -> None:
    output_dir = job.result_dir / "output"
    jsonl_path = output_dir / "kb_chunks.jsonl"
    required = [
        jsonl_path,
        output_dir / "kb_manifest.json",
        output_dir / "error_report.json",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise MinerUError(
            f"{job.source_pdf.name}: conversion output is incomplete; missing {', '.join(missing)}"
        )

    expected_fields = {
        "chunk_id",
        "page_no",
        "content_type",
        "section_title",
        "chunk_text",
        "image_path",
    }
    chunk_count = 0
    try:
        with jsonl_path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if not isinstance(chunk, dict):
                    raise MinerUError(
                        f"{job.source_pdf.name}: kb_chunks.jsonl line {line_no} is not an object"
                    )
                if set(chunk) != expected_fields:
                    raise MinerUError(
                        f"{job.source_pdf.name}: kb_chunks.jsonl fields on line {line_no} do not match the contract"
                    )
                if not str(chunk.get("chunk_text") or "").strip():
                    raise MinerUError(
                        f"{job.source_pdf.name}: kb_chunks.jsonl content on line {line_no} is empty"
                    )
                image_path = str(chunk.get("image_path") or "")
                if image_path:
                    image = Path(image_path)
                    if not image.is_absolute():
                        image = job.project_root / image
                    if not image.is_file():
                        raise MinerUError(
                            f"{job.source_pdf.name}: image path does not exist: {image_path}"
                        )
                chunk_count += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MinerUError(f"{job.source_pdf.name}: kb_chunks.jsonl could not be read") from exc
    if chunk_count == 0:
        raise MinerUError(f"{job.source_pdf.name}: kb_chunks.jsonl no valid chunks")

    error_report_path = output_dir / "error_report.json"
    try:
        with error_report_path.open("r", encoding="utf-8") as stream:
            error_report = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MinerUError(f"{job.source_pdf.name}: error_report.json could not be read") from exc
    if not isinstance(error_report, dict):
        raise MinerUError(f"{job.source_pdf.name}: error_report.json has an invalid format")

    unsupported = error_report.get("unsupported_blocks") or []
    parse_errors = error_report.get("parse_errors") or []
    missing_images = error_report.get("missing_images") or []
    oversized_tables = error_report.get("oversized_table_chunks") or []
    if unsupported:
        counts: dict[str, int] = {}
        for block in unsupported:
            block_type = (
                str(block.get("type") or "unknown")
                if isinstance(block, dict)
                else "unknown"
            )
            counts[block_type] = counts.get(block_type, 0) + 1
        details = ", ".join(
            f"{name}={count}" for name, count in sorted(counts.items())
        )
        raise MinerUError(f"{job.source_pdf.name}: conversion omitted unsupported block types: {details}")
    if parse_errors:
        raise MinerUError(
            f"{job.source_pdf.name}: conversion reported {len(parse_errors)} parse errors"
        )
    if missing_images:
        raise MinerUError(
            f"{job.source_pdf.name}: conversion reported {len(missing_images)} missing images"
        )
    if oversized_tables:
        raise MinerUError(
            f"{job.source_pdf.name}: conversion reported {len(oversized_tables)} oversized table chunks"
        )


def process_new_jobs(
    jobs: Sequence[Job],
    token: str,
    settings: dict[str, Any],
    poll_interval: float,
    poll_timeout: float,
) -> list[str]:
    failures = []
    for group in chunks(jobs, MAX_BATCH_FILES):
        print(f"requesting MinerU upload URLs: {len(group)} PDFs")
        batch_id, upload_urls = create_batch(group, token, settings)
        for job, upload_url in zip(group, upload_urls):
            print(f"[{job.source_pdf.name}] upload")
            upload_file(upload_url, job.source_pdf)

        results = poll_batch(
            batch_id,
            group,
            token,
            interval=poll_interval,
            timeout=poll_timeout,
        )
        for job in group:
            result = results[job.data_id]
            if result.get("state") == "failed":
                reason = result.get("err_msg") or "unknown reason"
                failures.append(f"{job.source_pdf.name}: MinerU parse failed: {reason}")
                continue
            try:
                install_result(job, result, batch_id, settings)
            except MinerUError as exc:
                message = str(exc)
                prefix = f"{job.source_pdf.name}:"
                failures.append(message if message.startswith(prefix) else f"{prefix} {message}")
    return failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="upload PDFs to MinerU and convert complete results to knowledge-base JSONL"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "one or more PDFs; directory input processes all *.pdf files in the top level; "
            "--convert-only may be omitted when used with --output-root"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="MinerU MinerU result root (default: each PDF's directory)",
    )
    parser.add_argument(
        "--shared-output",
        type=Path,
        help="additionally collect each document's knowledge-base JSONL into this directory",
    )
    parser.add_argument("--model", choices=("vlm", "pipeline"), default="vlm")
    parser.add_argument(
        "--language",
        default="en",
        help="document/OCR language, default en",
    )
    parser.add_argument("--ocr", action="store_true", help="force-enable OCR")
    parser.add_argument("--disable-table", action="store_true", help="disable table recognition")
    parser.add_argument("--disable-formula", action="store_true", help="disable formula recognition")
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="reuse existing *-mineru results; only reconvert and validate without accessing the MinerU API",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="poll interval in seconds, default 5",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=7200.0,
        help="maximum wait time per batch in seconds, default 7200",
    )
    return parser.parse_args(argv)

def parse_document_to_chunks(
    source: Path | list[Path],
    mineru_output_dir: Path,
    shared_output: Path | None = None,
    model: str = "vlm",
    language: str = "en",
    ocr: bool = False,
    enable_table: bool = True,
    enable_formula: bool = True,
    convert_only: bool = False,
    poll_interval: int = 5,
    timeout: int = 7200,
) -> None:
    """Parse documents with MinerU and convert the parsed content into KB chunks.
    Args:
        source: A document, directory, or list of documents to parse.
        mineru_output_dir: Directory for MinerU output.
        ...
    """

    try:
        if poll_interval <= 0 or timeout <= 0:
            raise MinerUError("poll_interval and timeout must be greater than 0")

        mineru_output_dir = mineru_output_dir.expanduser().resolve()
        shared_output = (
            shared_output.expanduser().resolve()
            if shared_output is not None
            else None
        )

        # Normalize source into a list of input paths.
        sources = [source] if isinstance(source, Path) else source

        if not sources:
            raise MinerUError("source must contain at least one document")

        # Convert Path objects to strings because expand_pdf_inputs()
        # currently accepts Sequence[str].
        inputs = [str(path) for path in sources]

        # A directory expands to all PDFs inside it.
        pdf_paths = expand_pdf_inputs(inputs)
        jobs = build_jobs(pdf_paths, mineru_output_dir)

        if not convert_only:
            mineru_output_dir.mkdir(parents=True, exist_ok=True)

        settings = {
            "model_version": model,
            "language": language,
            "is_ocr": ocr,
            "enable_table": enable_table,
            "enable_formula": enable_formula,
        }

        failures: list[str] = []

        if convert_only:
            installed: list[Job] = []

            for job in jobs:
                if not job.result_dir.is_dir():
                    failures.append(
                        f"{job.source_pdf.name}: "
                        f"MinerU result directory does not exist: {job.result_dir}"
                    )
                    continue

                try:
                    validate_result_dir(job.result_dir)
                    installed.append(job)
                except MinerUError as exc:
                    failures.append(f"{job.source_pdf.name}: {exc}")

        else:
            for job in jobs:
                if job.result_dir.exists():
                    raise MinerUError(
                        "Target directory already exists; refusing to overwrite or reuse: "
                        f"{job.result_dir}. Use --convert-only after a failed conversion."
                    )

            token = os.environ.get("MINERU_TOKEN", "").strip()

            if not token:
                raise MinerUError(
                    "MINERU_TOKEN is not set; configure this environment variable "
                    "before running."
                )

            failures.extend(
                process_new_jobs(
                    jobs,
                    token,
                    settings,
                    poll_interval,
                    timeout,
                )
            )

            installed = [
                job
                for job in jobs
                if job.result_dir.exists()
            ]

        completed: list[Job] = []

        for job in installed:
            try:
                run_converter(job, shared_output)
                completed.append(job)
            except MinerUError as exc:
                failures.append(str(exc))

        print("=" * 60)
        print(f"Completed: {len(completed)}/{len(jobs)} documents")

        for job in completed:
            print(f"  - {job.result_dir}/output/kb_chunks.jsonl")

        if failures:
            print("Failures:", file=sys.stderr)

            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)

    except (MinerUError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__" :
    cwd = Path.cwd()
    docs = cwd / "tests" / "documents"
    parse_document_to_chunks(source=docs, mineru_output_dir=cwd / "output" / "mineru")
