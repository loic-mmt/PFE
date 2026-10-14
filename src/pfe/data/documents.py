"""Small, immutable text documents and explicit, bounded collection.

Receipt is the default knowledge clock. SEC acceptance and fiscal dates never
stand in for publication. TXT/HTML use only the standard library; PDF text layers
require an already installed optional ``pypdf``. No OCR or model is downloaded.
"""

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from pfe.data.contracts import _instant
from pfe.data.http import BoundedHTTPTransport
from pfe.data.storage import RawStore, _json_bytes, _reject_secrets


def content_hash(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def utc_now():
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class DocumentLimits:
    max_document_bytes: int = 524_288
    max_text_bytes: int = 262_144
    max_total_bytes: int = 8_388_608
    max_documents: int = 20
    max_pdf_pages: int = 20

    def __post_init__(self):
        for value in asdict(self).values():
            if type(value) is not int or value < 1:
                raise ValueError("Document limits must be positive integers")


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.hidden += 1
        elif not self.hidden and tag in ("p", "div", "br", "tr", "li", "h1", "h2", "h3"):
            self.parts.append("\n")
        elif not self.hidden and tag in ("td", "th"):
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in ("p", "div", "tr", "li"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def parse_document(body, media_type, *, limits=None, encoding="utf-8"):
    """Return canonical text and PDF page offsets; fail on caps, never truncate."""
    limits = limits or DocumentLimits()
    if not isinstance(body, bytes) or len(body) > limits.max_document_bytes:
        raise ValueError("Document byte limit exceeded or non-bytes body")
    kind = media_type.split(";", 1)[0].lower().strip()
    pages = []
    if kind == "application/pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "PDF text import requires optional pypdf; use TXT/HTML instead"
            ) from None
        reader = PdfReader(io.BytesIO(body))
        if reader.is_encrypted or len(reader.pages) > limits.max_pdf_pages:
            raise ValueError("Encrypted PDF or PDF page limit exceeded")
        parts, offset = [], 0
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append({"page": number, "start": offset, "end": offset + len(text)})
            parts.append(text)
            offset += len(text) + 1
            if offset > limits.max_text_bytes:
                raise ValueError("Document text limit exceeded")
        text = "\n".join(parts)
    elif kind in ("text/plain", "text/html", "application/xhtml+xml"):
        text = body.decode(encoding, errors="strict").replace("\r\n", "\n").replace("\r", "\n")
        if kind != "text/plain":
            parser = _HTMLText()
            parser.feed(text)
            parser.close()
            text = "".join(parser.parts)
    else:
        raise ValueError("Only TXT, HTML and PDF text layers are supported")
    if len(text.encode("utf-8")) > limits.max_text_bytes:
        raise ValueError("Document text limit exceeded")
    if not text.strip():
        raise ValueError("Empty text layer; supply a text document (OCR is outside this importer)")
    return text, tuple(pages)


@dataclass(frozen=True)
class SourceDocument:
    source_document_id: str
    logical_id: str
    source_id: str
    text: str
    raw_sha256: str
    text_sha256: str
    revision_id: str
    received_at: str
    available_at: str
    published_at: str | None = None
    accepted_at: str | None = None
    entity_id: str | None = None
    observation_date: str | None = None
    url: str | None = None
    rights: str = "unknown"
    media_type: str = "text/plain"
    parser_version: str = "text-1"
    pages: tuple = ()
    collection_id: str | None = None

    def __post_init__(self):
        for name in ("source_document_id", "logical_id", "source_id", "text", "rights"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if content_hash(self.text) != self.text_sha256:
            raise ValueError("Document text hash mismatch")
        if not re.fullmatch(r"doc:[0-9a-f]{64}", self.source_document_id):
            raise ValueError("source_document_id must be a generated doc:SHA256 identity")
        if not re.fullmatch(r"[0-9a-f]{64}", self.raw_sha256):
            raise ValueError("Invalid raw document hash")
        received = _instant(self.received_at, "received_at")
        available = _instant(self.available_at, "available_at")
        if available < received:
            raise ValueError("Document availability cannot precede receipt")
        for field in ("published_at", "accepted_at"):
            value = getattr(self, field)
            if value is not None and _instant(value, field) > received:
                raise ValueError(f"{field} cannot follow document receipt")
        if self.observation_date is not None:
            from datetime import date

            if date.fromisoformat(self.observation_date).isoformat() != self.observation_date:
                raise ValueError("observation_date must be an ISO economic date")

    def to_dict(self):
        return {
            **asdict(self),
            "pages": list(self.pages),
            "availability_basis": "receipt",
            "point_in_time": True,
            "schema_version": "documents-1",
            "ingested_at": self.received_at,
        }

    @classmethod
    def from_dict(cls, record):
        fields = cls.__dataclass_fields__
        values = {name: value for name, value in record.items() if name in fields}
        values["pages"] = tuple(values.get("pages", ()))
        return cls(**values)


class DocumentStore:
    """Archive raw bytes once per content/metadata version with a per-run budget."""

    def __init__(self, root, *, limits=None):
        self.limits = limits or DocumentLimits()
        self.raw = RawStore(
            root,
            max_page_bytes=max(
                self.limits.max_document_bytes, self.limits.max_text_bytes * 6 + 16_384
            ),
        )
        self._bytes = self._count = self._transfers = self._downloaded = 0
        self._transports = {}

    def import_bytes(
        self,
        body,
        *,
        logical_id,
        source_id="local",
        media_type="text/plain",
        received_at=None,
        published_at=None,
        accepted_at=None,
        entity_id=None,
        observation_date=None,
        url=None,
        rights="unknown",
        encoding="utf-8",
    ):
        limits = self.limits
        text, pages = parse_document(body, media_type, limits=limits, encoding=encoding)
        identity = dict(
            logical_id=logical_id,
            source_id=source_id,
            raw_sha256=content_hash(body),
            text_sha256=content_hash(text),
            published_at=published_at,
            accepted_at=accepted_at,
            entity_id=entity_id,
            observation_date=observation_date,
            url=url,
            rights=rights,
            media_type=media_type,
            parser_version="text-1",
            encoding=encoding,
        )
        document_id = "doc:" + content_hash(_json_bytes(identity))
        receipt = received_at or utc_now()
        doc = SourceDocument(
            source_document_id=document_id,
            text=text,
            revision_id=content_hash(body),
            received_at=receipt,
            available_at=receipt,
            pages=pages,
            **{key: value for key, value in identity.items() if key != "encoding"},
        )
        # Check the budget before creating a manifest. Raw+parsed payloads both count.
        size = len(body) + len(_json_bytes(doc.to_dict())) + 512
        if self._count >= limits.max_documents or self._bytes + size > limits.max_total_bytes:
            raise ValueError("Document collection budget exceeded")
        collection = self.raw.begin(
            source_id,
            "source_documents",
            identity,
            source_metadata={"rights": rights},
            schema_version="documents-1",
        )
        manifest = self.raw.recover(collection)
        if manifest["complete"]:
            return self.load(collection)
        doc = SourceDocument.from_dict({**doc.to_dict(), "collection_id": collection})
        # Recover first receipt after interruption between raw and parsed pages.
        if manifest["pages"]:
            stored = manifest["pages"][0]["metadata"]["received_at"]
            doc = SourceDocument.from_dict(
                {**doc.to_dict(), "received_at": stored, "available_at": stored}
            )
        self.raw.write_page(
            collection, "raw", body, metadata={"received_at": doc.received_at}, row_count=0
        )
        self.raw.write_page(collection, "document", doc.to_dict(), row_count=1, complete=True)
        self._bytes += size
        self._count += 1
        return doc

    def import_file(self, path, **metadata):
        path = Path(path)
        kinds = {
            ".txt": "text/plain",
            ".html": "text/html",
            ".htm": "text/html",
            ".pdf": "application/pdf",
        }
        kind = metadata.pop("media_type", kinds.get(path.suffix.lower()))
        if kind is None:
            raise ValueError("Specify media_type for files other than TXT/HTML/PDF")
        # A single bounded read also handles concurrent growth without loading huge files.
        with path.open("rb") as stream:
            body = stream.read(self.limits.max_document_bytes + 1)
        return self.import_bytes(body, media_type=kind, **metadata)

    def load(self, collection_id):
        manifest = self.raw.manifest(collection_id)
        if not manifest["complete"]:
            raise ValueError("Document archive is incomplete")
        doc = SourceDocument.from_dict(json.loads(self.raw.read_page(collection_id, "document")))
        if content_hash(self.raw.read_page(collection_id, "raw")) != doc.raw_sha256:
            raise ValueError("Document raw hash mismatch")
        return doc

    def collect_url(
        self, url, *, allowed_hosts, transport=None, user_agent="pfe-documents/1", **metadata
    ):
        """Fetch only this URL; no crawling, redirects, attachments or linked files."""
        hosts = (allowed_hosts,) if isinstance(allowed_hosts, str) else tuple(allowed_hosts)
        parts = urlsplit(url)
        if (
            parts.scheme != "https"
            or parts.hostname not in hosts
            or parts.username is not None
            or parts.password is not None
            or parts.port not in (None, 443)
            or parts.fragment
        ):
            raise ValueError("Explicit HTTPS URL must use an allowed host without credentials")
        _reject_secrets({"url": url})
        if self._transfers >= self.limits.max_documents or self._count >= self.limits.max_documents:
            raise ValueError("Document collection budget exceeded")
        remaining = self.limits.max_total_bytes - self._bytes
        download_remaining = self.limits.max_total_bytes - self._downloaded
        if remaining <= 1024 or download_remaining < 1:
            raise ValueError("Document collection budget exceeded")
        if transport is None:
            key = (tuple(sorted(hosts)), user_agent)
            if key not in self._transports:
                self._transports[key] = BoundedHTTPTransport(
                    hosts, headers={"User-Agent": user_agent}, min_interval=0.15
                )
            transport = self._transports[key]
        http = transport
        cap = min(self.limits.max_document_bytes, remaining // 2, download_remaining)
        self._transfers += 1
        response = http.get(url, max_bytes=cap)
        if len(response.body) > cap:
            raise ValueError("Document response byte limit exceeded")
        self._downloaded += len(response.body)
        kind = metadata.pop("media_type", response.content_type or "text/plain")
        return self.import_bytes(
            response.body, received_at=response.received_at, url=url, media_type=kind, **metadata
        )

    def collect_sec(
        self, *, cik, accession, primary_document, user_agent, transport=None, **metadata
    ):
        """Collect one already selected filing, without downloading submissions/bulk ZIPs."""
        if not re.fullmatch(r"\d{1,10}", str(cik)) or int(cik) < 1:
            raise ValueError("Invalid SEC CIK")
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            raise ValueError("Invalid SEC accession")
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", primary_document)
            or ".." in primary_document
        ):
            raise ValueError("SEC primary_document must be a filename")
        if not isinstance(user_agent, str) or "@" not in user_agent:
            raise ValueError("SEC requires an identified User-Agent with contact email")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession.replace('-', '')}/{primary_document}"
        )
        metadata.setdefault("logical_id", f"sec:{int(cik)}:{accession}:{primary_document}")
        metadata.setdefault("source_id", "sec")
        metadata.setdefault("rights", "SEC public filing; verify third-party exhibits rights")
        return self.collect_url(
            url,
            allowed_hosts=("www.sec.gov",),
            user_agent=user_agent,
            transport=transport,
            **metadata,
        )


def documents_frame(documents):
    """Metadata table; archived text/raw remain addressable through collection_id."""
    rows = [
        {
            **{key: value for key, value in doc.to_dict().items() if key != "text"},
            "value": {
                "collection_id": doc.collection_id,
                "text_sha256": doc.text_sha256,
                "raw_sha256": doc.raw_sha256,
            },
        }
        for doc in documents
    ]
    columns = [name for name in SourceDocument.__dataclass_fields__ if name != "text"] + [
        "value",
        "schema_version",
        "availability_basis",
        "point_in_time",
        "ingested_at",
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
