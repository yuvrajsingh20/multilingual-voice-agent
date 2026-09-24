"""Download the pinned regulatory corpus and record hashes.

Re-run this when RBI or MeitY publishes a change. It does not edit policy
rules. If a file hash changes, the previous copy is kept and the catalog
marks that document as needing review.

RBI's PDF host (rbidocs.rbi.org.in) often answers this network with a bot
challenge instead of the PDF. In that case the snapshot is the official
notification or master-direction HTML page, which carries the same text.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
SOURCES_PATH = ROOT / "sources.json"
CATALOG_PATH = ROOT / "catalog.json"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def is_pdf(data: bytes) -> bool:
    return data.lstrip().startswith(b"%PDF-")


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
        # Master-direction pages sometimes use a hyphen where the title uses a dash.
        loose = title.replace("–", "-").replace("—", "-")
        start = text.find(loose)
    if start > 0:
        text = text[start:]
    for marker in ("Best viewed in", "Website owned and managed by"):
        end = text.find(marker)
        if end > 500:
            text = text[:end]
    # RBI pages append the year/month archive navigator after the instrument.
    nav = re.search(r"\nAll Months\n", text)
    if nav and nav.start() > 500:
        text = text[: nav.start()]
    return text.strip() + "\n"


def extract_pdf_text(pdf_path: Path, text_path: Path) -> None:
    subprocess.run(
        ["pdftotext", "-enc", "UTF-8", "-layout", str(pdf_path), str(text_path)],
        check=True,
    )


def main() -> None:
    manifest = json.loads(SOURCES_PATH.read_text())
    previous = {}
    if CATALOG_PATH.exists():
        previous = {
            item["doc_id"]: item
            for item in json.loads(CATALOG_PATH.read_text()).get("documents", [])
        }

    documents = []
    for source in manifest["sources"]:
        doc_id = source["doc_id"]
        authority = source["authority"].lower()
        raw_dir = ROOT / "raw" / authority
        text_dir = ROOT / "text" / authority
        raw_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)

        file_bytes = fetch(source["file_url"])
        pdf_status = "downloaded"
        if is_pdf(file_bytes):
            raw_path = raw_dir / f"{doc_id}.pdf"
            text_path = text_dir / f"{doc_id}.txt"
            snapshot = file_bytes
            raw_format = "pdf"
        else:
            page_bytes = fetch(source["source_page"])
            if is_pdf(page_bytes):
                raw_path = raw_dir / f"{doc_id}.pdf"
                snapshot = page_bytes
                raw_format = "pdf"
                pdf_status = "downloaded_from_source_page"
            else:
                raw_path = raw_dir / f"{doc_id}.html"
                snapshot = page_bytes
                raw_format = "html"
                pdf_status = "pdf_host_blocked_saved_official_html"
            text_path = text_dir / f"{doc_id}.txt"

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
                "archived_raw": str(archive.relative_to(ROOT)),
            }

        raw_path.write_bytes(snapshot)
        if raw_format == "pdf":
            extract_pdf_text(raw_path, text_path)
        else:
            text_path.write_text(
                html_to_text(snapshot.decode("utf-8", "replace"), source["title"]),
                encoding="utf-8",
            )

        record = {
            **source,
            "raw_format": raw_format,
            "pdf_status": pdf_status,
            "local_raw": str(raw_path.relative_to(ROOT)),
            "local_text": str(text_path.relative_to(ROOT)),
            "sha256": digest,
            "bytes": len(snapshot),
            "text_bytes": text_path.stat().st_size,
        }
        if review:
            record["refresh_review"] = review
        documents.append(record)
        print(f"{doc_id} {raw_format} {len(snapshot)} text={text_path.stat().st_size}")

    catalog = {
        "corpus_id": manifest["corpus_id"],
        "corpus_version": manifest["corpus_version"],
        "ingested_on": manifest["ingested_on"],
        "refreshed_on": date.today().isoformat(),
        "purpose": manifest["purpose"],
        "documents": documents,
    }
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {CATALOG_PATH}")


if __name__ == "__main__":
    main()
