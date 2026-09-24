#!/usr/bin/env python3
"""Download bank-specific RBI source documents into data/regulatory/raw/.

Reads the pin list at data/regulatory/sources.json. Writes:
  data/regulatory/raw/<id>.{html|pdf}
  data/regulatory/text/<id>.txt
  data/regulatory/catalog.json

Does not extract or encode policy rules. The policy engine does that later.

RBI's PDF host (rbidocs.rbi.org.in / some commonman PDFs) may answer with a
bot challenge. When that happens, the official notification HTML page is saved
instead — it carries the same instrument text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = REPO_ROOT / "data" / "regulatory" / "sources.json"
RAW_DIR = REPO_ROOT / "data" / "regulatory" / "raw"
TEXT_DIR = REPO_ROOT / "data" / "regulatory" / "text"
CATALOG_PATH = REPO_ROOT / "data" / "regulatory" / "catalog.json"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

SKIP_STATUSES = {"out_of_scope"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_pdf(data: bytes) -> bool:
    return data.lstrip().startswith(b"%PDF-")


def looks_like_challenge(data: bytes) -> bool:
    head = data[:2000].lower()
    return b"captcha" in head or b"cf-challenge" in head or b"attention required" in head


def fetch(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
            "Referer": "https://www.rbi.org.in/",
        },
    )
    with urlopen(request, timeout=120) as response:
        return response.read()


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        if tag in {"p", "br", "tr", "h1", "h2", "h3", "li", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        if tag in {"p", "tr", "h1", "h2", "h3", "li", "table"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip == 0:
            self.parts.append(data)


def html_to_text(html: str, title: str) -> str:
    parser = _VisibleText()
    parser.feed(html)
    text = "".join(parser.parts)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    start = text.find(title)
    if start == -1:
        loose = title.replace("–", "-").replace("—", "-")
        start = text.find(loose)
    if start > 0:
        text = text[start:]
    for marker in ("Best viewed in", "Website owned and managed by"):
        end = text.find(marker)
        if end > 500:
            text = text[:end]
    nav = re.search(r"\nAll Months\n", text)
    if nav and nav.start() > 500:
        text = text[: nav.start()]
    return text.strip() + "\n"


def extract_pdf_text(pdf_path: Path, text_path: Path) -> None:
    subprocess.run(
        ["pdftotext", "-enc", "UTF-8", "-layout", str(pdf_path), str(text_path)],
        check=True,
    )


def candidate_urls(source: dict) -> list[str]:
    urls: list[str] = []
    for key in ("file_url", "url"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            urls.append(value.strip())
    companions = source.get("companion_urls") or []
    if isinstance(companions, list):
        for value in companions:
            if isinstance(value, str) and value.strip():
                urls.append(value.strip())
    # Prefer direct PDFs first, then official HTML pages.
    urls.sort(key=lambda u: (0 if u.lower().endswith(".pdf") else 1, u))
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def download_source(source: dict) -> tuple[bytes, str, str]:
    """Return (bytes, raw_format, fetch_status)."""
    urls = candidate_urls(source)
    if not urls:
        raise ValueError("no downloadable URL")

    errors: list[str] = []
    html_fallback: tuple[bytes, str] | None = None

    for url in urls:
        try:
            data = fetch(url)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
            continue

        if is_pdf(data) and not looks_like_challenge(data):
            return data, "pdf", f"downloaded:{url}"

        if looks_like_challenge(data):
            errors.append(f"{url}: bot challenge")
            continue

        # Keep the first usable HTML page as fallback.
        if html_fallback is None and data.strip():
            html_fallback = (data, url)

    if html_fallback is not None:
        data, url = html_fallback
        return data, "html", f"saved_official_html:{url}"

    raise RuntimeError("; ".join(errors) or "download failed")


def write_text(raw_path: Path, text_path: Path, raw_format: str, title: str) -> None:
    if raw_format == "pdf":
        extract_pdf_text(raw_path, text_path)
        return
    html = raw_path.read_text(encoding="utf-8", errors="replace")
    text_path.write_text(html_to_text(html, title), encoding="utf-8")


def ingest_one(source: dict, previous: dict[str, dict]) -> dict:
    doc_id = source["id"]
    title = source.get("title") or doc_id
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    snapshot, raw_format, fetch_status = download_source(source)
    raw_path = RAW_DIR / f"{doc_id}.{raw_format}"
    text_path = TEXT_DIR / f"{doc_id}.txt"

    digest = sha256(snapshot)
    prior = previous.get(doc_id)
    review = None
    if prior and prior.get("sha256") and prior["sha256"] != digest:
        archive = raw_path.with_name(
            f"{raw_path.stem}.before-{date.today().isoformat()}{raw_path.suffix}"
        )
        if raw_path.exists():
            shutil.copy2(raw_path, archive)
        review = {
            "status": "source_changed_review_required",
            "previous_sha256": prior["sha256"],
            "archived_raw": str(archive.relative_to(REPO_ROOT)),
        }

    raw_path.write_bytes(snapshot)
    write_text(raw_path, text_path, raw_format, title)

    record = {
        **source,
        "raw_format": raw_format,
        "fetch_status": fetch_status,
        "local_raw": str(raw_path.relative_to(REPO_ROOT)),
        "local_text": str(text_path.relative_to(REPO_ROOT)),
        "sha256": digest,
        "bytes": len(snapshot),
        "text_bytes": text_path.stat().st_size,
        "ingested_on": date.today().isoformat(),
    }
    if review:
        record["refresh_review"] = review
    return record


def load_previous_catalog() -> dict[str, dict]:
    if not CATALOG_PATH.exists():
        return {}
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {item["id"]: item for item in data.get("documents", []) if "id" in item}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest bank-specific RBI regulatory sources into data/regulatory/raw/."
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=DEFAULT_SOURCES,
        help="Path to sources.json pin list (default: data/regulatory/sources.json)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional source id(s) to ingest; default is all downloadable entries",
    )
    args = parser.parse_args(argv)

    sources_path = args.sources.resolve()
    if not sources_path.exists():
        print(f"missing sources file: {sources_path}", file=sys.stderr)
        return 1

    manifest = json.loads(sources_path.read_text(encoding="utf-8"))
    previous = load_previous_catalog()
    documents: list[dict] = []
    skipped: list[dict] = []
    failed: list[dict] = []

    for source in manifest.get("sources", []):
        doc_id = source.get("id")
        if not doc_id:
            skipped.append({"reason": "missing_id", "source": source})
            continue
        if args.only is not None and doc_id not in args.only:
            continue
        if source.get("status") in SKIP_STATUSES:
            skipped.append({"id": doc_id, "reason": "out_of_scope"})
            continue
        if not candidate_urls(source):
            skipped.append({"id": doc_id, "reason": "no_url"})
            print(f"SKIP {doc_id} (no URL)")
            continue

        try:
            record = ingest_one(source, previous)
        except Exception as exc:  # noqa: BLE001 — surface per-doc failure, keep going
            failed.append({"id": doc_id, "error": str(exc)})
            print(f"FAIL {doc_id}: {exc}", file=sys.stderr)
            continue

        documents.append(record)
        print(
            f"OK   {doc_id} {record['raw_format']} "
            f"{record['bytes']}B text={record['text_bytes']}B"
        )

    catalog = {
        "corpus_id": "commercial-bank-debt-recovery-voice-agent",
        "as_of": manifest.get("as_of"),
        "product_scope": manifest.get("product_scope"),
        "sources_file": str(sources_path.relative_to(REPO_ROOT)),
        "ingested_on": date.today().isoformat(),
        "note": (
            "Raw snapshots only. No policy rules are encoded here. "
            "The 2022 recovery-agent circular supplements existing RBI bank "
            "recovery/outsourcing instructions; both layers are retained in the pin list."
        ),
        "documents": documents,
        "skipped": skipped,
        "failed": failed,
    }
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {CATALOG_PATH.relative_to(REPO_ROOT)}")
    print(f"ingested={len(documents)} skipped={len(skipped)} failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
