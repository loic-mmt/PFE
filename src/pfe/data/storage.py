"""Immutable raw pages and atomic, resumable collection manifests.

Collection identity includes the request and source/license revision. A changed
historical snapshot must use a new request (for example, a new ``snapshot_at``).
JSON pages are canonicalized; bytes are preserved exactly. Pagination cursors
are opaque continuation state, never credentials.
"""

import errno
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

_STORAGE_VERSION = 1
_COLLECTION_ID = re.compile(r"[0-9a-f]{64}\Z")
_PAGE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_PAGINATION_KEYS = {"pagetoken", "nextpagetoken", "paginationtoken", "continuationtoken"}
_SECRET_KEYS = {
    "auth",
    "authorization",
    "credential",
    "credentials",
    "cookie",
    "setcookie",
    "apikey",
    "accesskey",
    "secretkey",
    "privatekey",
    "sessionkey",
    "token",
    "password",
    "passwd",
    "secret",
}
_PARTIAL_FLAGS = {
    "truncated",
    "capped",
    "cap_reached",
    "row_cap_reached",
    "byte_cap_reached",
    "partial",
    "incomplete",
    "has_more",
}


class RawStoreError(ValueError):
    """Invalid or inconsistent raw storage operation."""


class ImmutablePageError(RawStoreError):
    """An existing page cannot be replaced by different data or metadata."""


class CorruptStoreError(RawStoreError):
    """Stored identity, page contents, or manifest no longer match."""


def _json_bytes(value: Any) -> bytes:
    """Canonical JSON, rejecting non-string keys and non-finite numbers."""

    def validate(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise RawStoreError("JSON object keys must be strings")
            for nested in item.values():
                validate(nested)
        elif isinstance(item, list):
            for nested in item:
                validate(nested)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise RawStoreError("Values must be JSON-compatible")

    try:
        validate(value)
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise RawStoreError("Values must be finite, valid JSON") from exc


def _clone(value: Any) -> Any:
    return json.loads(_json_bytes(value))


def _secret_key(key: str) -> bool:
    normalized = "".join(char for char in key.lower() if char.isalnum())
    if normalized in _PAGINATION_KEYS:
        return False
    return normalized in _SECRET_KEYS or normalized.endswith(
        ("apikey", "password", "passwd", "secret", "token", "authorization")
    )


def _reject_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if _secret_key(key):
                raise RawStoreError("Credentials must not be persisted in requests or metadata")
            _reject_secrets(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_secrets(nested)
    elif isinstance(value, str) and "://" in value:
        try:
            url = urlsplit(value)
            credential = url.username is not None or url.password is not None
            credential |= any(_secret_key(key) for key, _ in parse_qsl(url.query))
        except ValueError as exc:
            raise RawStoreError("Invalid URL in request or metadata") from exc
        if credential:
            raise RawStoreError("Credentials must not be persisted in URLs")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RawStoreError(f"{label} must be a JSON object")
    result = _clone(value)
    _reject_secrets(result)
    return result


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EBADF):
                raise
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, body: bytes, *, replace: bool) -> bool:
    """Fsync a same-directory temporary file, then publish atomically.

    Hard linking implements no-clobber publication. Unsupported filesystems fail
    safely instead of falling back to a partially visible direct write.
    """
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
        _fsync_directory(path.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise CorruptStoreError("Stored file is missing or not a regular file")


def _load_json(path: Path) -> dict[str, Any]:
    _regular_file(path)
    try:
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, UnicodeError) as exc:
        raise CorruptStoreError("Invalid stored JSON") from exc


@contextmanager
def _locked(directory: Path):
    """Serialize both threads and processes; OS locks release after a crash."""
    key = str(directory)
    with _THREAD_LOCKS_GUARD:
        local_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
    with local_lock:
        lock_path = directory / ".lock"
        if lock_path.is_symlink():
            raise CorruptStoreError("Lock file must not be a symlink")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                deadline = time.monotonic() + 30
                while True:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    try:
                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Timed out acquiring raw-store lock") from None
                        time.sleep(0.01)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class RawStore:
    """Store immutable raw pages under deterministic collection IDs.

    Completion is explicit. A complete collection only accepts identical page
    replays. A failed manifest commit leaves an immutable page record/raw file;
    call ``recover`` when the raw bytes were published, or retry the identical
    ``write_page`` when publication stopped before the raw bytes were stored.
    """

    def __init__(self, root: str | Path, max_page_bytes: int = 1_048_576):
        if type(max_page_bytes) is not int or max_page_bytes <= 0:
            raise RawStoreError("max_page_bytes must be a positive integer")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_page_bytes = max_page_bytes

    def _directory(self, collection_id: str, *, create: bool = False) -> Path:
        if not isinstance(collection_id, str) or not _COLLECTION_ID.fullmatch(collection_id):
            raise RawStoreError("Invalid collection_id")
        directory = self.root / collection_id
        if directory.is_symlink():
            raise CorruptStoreError("Collection directory must not be a symlink")
        if create:
            directory.mkdir(exist_ok=True, mode=0o700)
        if not directory.is_dir():
            raise KeyError("Unknown collection_id")
        pages = directory / "pages"
        if pages.is_symlink():
            raise CorruptStoreError("Pages directory must not be a symlink")
        if create:
            pages.mkdir(exist_ok=True, mode=0o700)
        return directory

    @staticmethod
    def _page_paths(directory: Path, page_id: str) -> tuple[Path, Path]:
        if not isinstance(page_id, str) or not _PAGE_ID.fullmatch(page_id) or ".." in page_id:
            raise RawStoreError("Invalid page_id")
        stem = _sha256(page_id.encode("utf-8"))
        return directory / "pages" / f"{stem}.bin", directory / "pages" / f"{stem}.json"

    @staticmethod
    def _identity(manifest: dict[str, Any]) -> dict[str, Any]:
        return {
            key: manifest[key]
            for key in ("source_id", "dataset", "request", "source_metadata", "schema_version")
        }

    def begin(
        self,
        source_id: str,
        dataset: str,
        request: dict[str, Any],
        *,
        source_metadata: dict[str, Any] | None = None,
        schema_version: str = "1",
    ) -> str:
        """Create or replay a collection; preserve source/license metadata."""
        for label, value in (
            ("source_id", source_id),
            ("dataset", dataset),
            ("schema_version", schema_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise RawStoreError(f"{label} must be a non-empty string")
        identity = {
            "source_id": source_id,
            "dataset": dataset,
            "request": _object(request, "request"),
            "source_metadata": _object(
                {} if source_metadata is None else source_metadata, "source_metadata"
            ),
            "schema_version": schema_version,
        }
        collection_id = _sha256(_json_bytes(identity))
        directory = self._directory(collection_id, create=True)
        with _locked(directory):
            path = directory / "manifest.json"
            if path.exists() or path.is_symlink():
                self._manifest(directory, collection_id)
                return collection_id
            receipt = _timestamp()
            manifest = {
                **identity,
                "storage_version": _STORAGE_VERSION,
                "collection_id": collection_id,
                "created_at": receipt,
                "updated_at": receipt,
                "complete": False,
                "next_cursor": None,
                "page_count": 0,
                "row_count": 0,
                "known_row_count": 0,
                "pages": [],
            }
            _atomic_write(path, _json_bytes(manifest), replace=False)
        return collection_id

    def _manifest(
        self, directory: Path, collection_id: str, *, verify_raw: bool = True
    ) -> dict[str, Any]:
        path = directory / "manifest.json"
        if not path.exists() and not path.is_symlink():
            raise KeyError("Unknown collection_id")
        manifest = _load_json(path)
        try:
            identity_hash = _sha256(_json_bytes(self._identity(manifest)))
            pages = manifest["pages"]
            valid = (
                identity_hash == collection_id == manifest["collection_id"]
                and manifest["storage_version"] == _STORAGE_VERSION
                and isinstance(pages, list)
                and manifest["page_count"] == len(pages)
            )
            if not valid:
                raise CorruptStoreError("Invalid manifest identity or page count")
            seen: set[str] = set()
            for index, page in enumerate(pages):
                self._validate_record(page, collection_id, index)
                if page["page_id"] in seen or page["sequence"] != index:
                    raise CorruptStoreError("Duplicate or unordered manifest page")
                seen.add(page["page_id"])
                raw_path, record_path = self._page_paths(directory, page["page_id"])
                if _load_json(record_path) != page:
                    raise CorruptStoreError("Manifest page does not match immutable record")
                if verify_raw:
                    self._verify_raw(raw_path, page)
                else:
                    _regular_file(raw_path)
                if index < len(pages) - 1 and page["complete"]:
                    raise CorruptStoreError("Pages follow a completed collection")
            self._check_state(manifest)
        except (KeyError, TypeError) as exc:
            raise CorruptStoreError("Missing or invalid manifest fields") from exc
        return manifest

    @staticmethod
    def _validate_record(page: dict[str, Any], collection_id: str, sequence: int) -> None:
        if not isinstance(page, dict):
            raise CorruptStoreError("Invalid immutable page record")
        rows = page.get("row_count")
        size = page.get("size_bytes")
        cursor = page.get("next_cursor")
        checksum = page.get("sha256")
        receipt = page.get("received_at")
        valid = (
            page.get("collection_id") == collection_id
            and page.get("storage_version") == _STORAGE_VERSION
            and type(page.get("sequence")) is int
            and page["sequence"] == sequence
            and type(page.get("complete")) is bool
            and type(size) is int
            and size >= 0
            and (rows is None or (type(rows) is int and rows >= 0))
            and (cursor is None or (isinstance(cursor, str) and bool(cursor)))
            and not (page.get("complete") and cursor is not None)
            and isinstance(checksum, str)
            and _COLLECTION_ID.fullmatch(checksum) is not None
            and page.get("payload_format") in ("bytes", "json")
            and isinstance(page.get("metadata"), dict)
            and isinstance(receipt, str)
            and receipt.endswith("Z")
        )
        if not valid:
            raise CorruptStoreError("Invalid immutable page record")
        try:
            datetime.fromisoformat(receipt)
        except ValueError as exc:
            raise CorruptStoreError("Invalid page receipt timestamp") from exc

    @staticmethod
    def _verify_raw(path: Path, page: dict[str, Any]) -> bytes:
        _regular_file(path)
        body = path.read_bytes()
        if len(body) != page["size_bytes"] or _sha256(body) != page["sha256"]:
            raise CorruptStoreError("Raw page checksum mismatch")
        return body

    @staticmethod
    def _check_state(manifest: dict[str, Any]) -> None:
        pages = manifest["pages"]
        known_rows = sum(page["row_count"] or 0 for page in pages)
        row_count = known_rows if all(page["row_count"] is not None for page in pages) else None
        if (
            manifest["known_row_count"] != known_rows
            or manifest["row_count"] != row_count
            or manifest["next_cursor"] != (pages[-1]["next_cursor"] if pages else None)
            or manifest["complete"] != (pages[-1]["complete"] if pages else False)
        ):
            raise CorruptStoreError("Manifest pagination or row totals mismatch")

    def manifest(self, collection_id: str) -> dict[str, Any]:
        """Return a detached manifest after verifying referenced page checksums."""
        directory = self._directory(collection_id)
        with _locked(directory):
            return self._manifest(directory, collection_id)

    def recover(self, collection_id: str) -> dict[str, Any]:
        """Commit a journaled raw page without fetching or changing its vintage.

        Recovery is idempotent and preserves the complete immutable page record,
        including receipt time and pagination state. Missing or corrupt pending
        raw bytes cannot be reconstructed: retry the original ``write_page``
        input, or start a new snapshot rather than replacing that vintage.
        """
        directory = self._directory(collection_id)
        with _locked(directory):
            manifest = self._manifest(directory, collection_id)
            referenced = {self._page_paths(directory, p["page_id"])[1] for p in manifest["pages"]}
            pending = set((directory / "pages").glob("*.json")) - referenced
            if not pending:
                return manifest
            if len(pending) != 1 or manifest["complete"]:
                raise CorruptStoreError("Ambiguous pending pages or pending completed collection")
            record_path = next(iter(pending))
            page = _load_json(record_path)
            self._validate_record(page, collection_id, len(manifest["pages"]))
            try:
                raw_path, expected_record = self._page_paths(directory, page["page_id"])
            except (KeyError, RawStoreError) as exc:
                raise CorruptStoreError("Invalid pending page identity") from exc
            if record_path != expected_record or any(
                p["page_id"] == page["page_id"] for p in manifest["pages"]
            ):
                raise CorruptStoreError("Pending page identity does not match its journal")
            if not raw_path.exists() and not raw_path.is_symlink():
                raise CorruptStoreError(
                    "Pending raw page is missing; retry the original write_page"
                )
            self._verify_raw(raw_path, page)
            self._append_page(manifest, page)
            _atomic_write(directory / "manifest.json", _json_bytes(manifest), replace=True)
            return manifest

    @staticmethod
    def _append_page(manifest: dict[str, Any], page: dict[str, Any]) -> None:
        manifest["pages"].append(page)
        manifest["page_count"] = len(manifest["pages"])
        manifest["known_row_count"] += page["row_count"] or 0
        manifest["row_count"] = (
            manifest["known_row_count"]
            if all(p["row_count"] is not None for p in manifest["pages"])
            else None
        )
        manifest["next_cursor"] = page["next_cursor"]
        manifest["complete"] = page["complete"]
        manifest["updated_at"] = _timestamp()

    def write_page(
        self,
        collection_id: str,
        page_id: str,
        payload: dict[str, Any] | list[Any] | bytes,
        *,
        next_cursor: str | None = None,
        complete: bool = False,
        row_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a page or replay it identically; return its immutable record."""
        if isinstance(payload, bytes):
            body, payload_format = payload, "bytes"
        elif isinstance(payload, (dict, list)):
            body, payload_format = _json_bytes(payload), "json"
        else:
            raise RawStoreError("payload must be a JSON object/array or bytes")
        if len(body) > self.max_page_bytes:
            raise RawStoreError("Page exceeds max_page_bytes; collection remains incomplete")
        if type(complete) is not bool:
            raise RawStoreError("complete must be a boolean")
        if next_cursor is not None and (not isinstance(next_cursor, str) or not next_cursor):
            raise RawStoreError("next_cursor must be a non-empty string or None")
        if row_count is not None and (type(row_count) is not int or row_count < 0):
            raise RawStoreError("row_count must be a non-negative integer or None")
        page_metadata = _object({} if metadata is None else metadata, "metadata")
        if complete and (
            next_cursor is not None
            or any(page_metadata.get(flag) is True for flag in _PARTIAL_FLAGS)
        ):
            raise RawStoreError("A capped or continuing page cannot complete the collection")
        directory = self._directory(collection_id)
        raw_path, record_path = self._page_paths(directory, page_id)
        expected = {
            "collection_id": collection_id,
            "page_id": page_id,
            "sha256": _sha256(body),
            "size_bytes": len(body),
            "row_count": row_count,
            "next_cursor": next_cursor,
            "complete": complete,
            "metadata": page_metadata,
        }
        with _locked(directory):
            manifest = self._manifest(directory, collection_id, verify_raw=False)
            existing = next((p for p in manifest["pages"] if p["page_id"] == page_id), None)
            if existing is not None:
                self._verify_raw(raw_path, existing)
                self._same_page(existing, expected)
                return existing
            if manifest["complete"]:
                raise ImmutablePageError("Complete collection requires a new snapshot request")
            referenced = {self._page_paths(directory, p["page_id"])[1] for p in manifest["pages"]}
            pending = set((directory / "pages").glob("*.json")) - referenced
            if pending - {record_path}:
                raise RawStoreError("Recover the pending page before appending another page")
            if record_path.exists() or record_path.is_symlink():
                page = _load_json(record_path)
                self._validate_record(page, collection_id, len(manifest["pages"]))
                self._same_page(page, expected)
                if page.get("sequence") != len(manifest["pages"]):
                    raise CorruptStoreError("Pending page sequence mismatch")
            else:
                if raw_path.exists() or raw_path.is_symlink():
                    raise CorruptStoreError("Raw page has no immutable metadata record")
                page = {
                    **expected,
                    "storage_version": _STORAGE_VERSION,
                    "sequence": len(manifest["pages"]),
                    "payload_format": payload_format,
                    "received_at": _timestamp(),
                }
                _atomic_write(record_path, _json_bytes(page), replace=False)
            if not _atomic_write(raw_path, body, replace=False):
                self._verify_raw(raw_path, page)
            self._append_page(manifest, page)
            _atomic_write(directory / "manifest.json", _json_bytes(manifest), replace=True)
            return page

    @staticmethod
    def _same_page(page: dict[str, Any], expected: dict[str, Any]) -> None:
        if any(page.get(key) != value for key, value in expected.items()):
            raise ImmutablePageError("Page body or metadata conflicts with immutable original")

    def read_page(self, collection_id: str, page_id: str) -> bytes:
        """Read only a committed page, verifying its checksum before return."""
        directory = self._directory(collection_id)
        raw_path, _ = self._page_paths(directory, page_id)
        with _locked(directory):
            manifest = self._manifest(directory, collection_id, verify_raw=False)
            page = next((p for p in manifest["pages"] if p["page_id"] == page_id), None)
            if page is None:
                raise KeyError("Unknown page_id")
            return self._verify_raw(raw_path, page)

    def resume_cursor(self, collection_id: str) -> str | None:
        """Return continuation cursor; check ``manifest()['complete']`` separately."""
        return self.manifest(collection_id)["next_cursor"]
