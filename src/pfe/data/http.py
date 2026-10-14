"""Bounded, GET-only provider transport; adapters own schema and availability policy."""

import http.client
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime

DEFAULT_PROVIDER_HOSTS = frozenset(
    (
        "api.stlouisfed.org",
        "data.sec.gov",
        "www.sec.gov",
        "api.eia.gov",
        "cdn.cboe.com",
        "cdn-api.cboe.com",
        "www.cboe.com",
        "thedocs.worldbank.org",
        "www.worldbank.org",
    )
)


class HTTPTransportError(ValueError):
    """Provider request failure with no credentials, response body or request URL."""


@dataclass(frozen=True)
class HTTPResponse:
    body: bytes
    received_at: str
    content_type: str | None = None

    def __post_init__(self):
        if not isinstance(self.body, bytes):
            raise ValueError("HTTP response body must be bytes")
        try:
            instant = datetime.fromisoformat(self.received_at)
        except (TypeError, ValueError):
            raise ValueError("HTTP receipt must be an aware timestamp") from None
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("HTTP receipt must be an aware timestamp")
        if self.content_type is not None and not isinstance(self.content_type, str):
            raise ValueError("content_type must be string or null")

    @property
    def byte_count(self):
        return len(self.body)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _headers(value):
    if not isinstance(value, dict):
        raise ValueError("HTTP headers require an object")
    for name, text in value.items():
        if not isinstance(name, str) or not isinstance(text, str) or not name or not text:
            raise ValueError("HTTP header names and values must be nonempty strings")
        if any(ord(char) < 32 or ord(char) == 127 for char in name + text):
            raise ValueError("HTTP headers must not contain control characters")
    return value.copy()


class BoundedHTTPTransport:
    """One capped HTTPS GET per call, no automatic retry or redirects.

    Query credentials required by FRED/EIA are sent only in the actual request;
    this transport never includes URLs, params or provider error bodies in its
    errors. JSON adapters must remove provider echoes of credentials before raw
    persistence. Binary/CSV bodies remain exact evidence. min_interval throttles
    starts across threads sharing this instance, using a monotonic clock.
    """

    def __init__(
        self,
        allowed_hosts=None,
        *,
        headers=None,
        min_interval=0,
        timeout=30,
        opener=None,
    ):
        hosts = DEFAULT_PROVIDER_HOSTS if allowed_hosts is None else allowed_hosts
        if isinstance(hosts, str):
            hosts = (hosts,)
        self._hosts = frozenset(hosts)
        if not self._hosts or any(
            not isinstance(host, str)
            or not host
            or host != host.lower()
            or any(char in host for char in "/:@?# ")
            for host in self._hosts
        ):
            raise ValueError("allowed_hosts must contain explicit lowercase hostnames")
        for name, number in (("min_interval", min_interval), ("timeout", timeout)):
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= min_interval <= 60 or not 0 < timeout <= 60:
            raise ValueError("Require 0 <= min_interval <= 60 and 0 < timeout <= 60")
        self._headers = _headers({"User-Agent": "pfe-data/1"} if headers is None else headers)
        self._interval, self._timeout = min_interval, timeout
        self._opener = opener or urllib.request.build_opener(_NoRedirect())
        self._lock, self._last_start = threading.Lock(), None

    def get(self, url, *, params=None, max_bytes, headers=None):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("HTTP max_bytes must be a positive integer")
        if not isinstance(url, str):
            raise ValueError("Provider URL must be an HTTPS string")
        try:
            parsed = urllib.parse.urlsplit(url)
            valid = (
                parsed.scheme == "https"
                and parsed.hostname in self._hosts
                and parsed.username is None
                and parsed.password is None
                and parsed.port in (None, 443)
                and not parsed.fragment
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(
                "Provider URL requires HTTPS and an allowed host, without userinfo/fragment"
            )
        if params is not None:
            if not isinstance(params, dict):
                raise ValueError("HTTP params must be an object")
            for name, value in params.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("HTTP query names must be strings")
                values = value if isinstance(value, (list, tuple)) else (value,)
                if any(type(item) not in (str, int, float, type(None)) for item in values):
                    raise ValueError("HTTP query values must be strings or finite numbers")
                if any(isinstance(item, float) and not math.isfinite(item) for item in values):
                    raise ValueError("HTTP query values must be finite")
            query = urllib.parse.urlencode(
                {name: value for name, value in params.items() if value is not None}, doseq=True
            )
            url += ("&" if parsed.query else "?") + query
        request_headers = self._headers.copy()
        if headers is not None:
            request_headers.update(_headers(headers))
        try:
            request = urllib.request.Request(url, headers=request_headers, method="GET")
            with self._lock:
                if self._last_start is not None:
                    delay = self._interval - (time.monotonic() - self._last_start)
                    if delay > 0:
                        time.sleep(delay)
                self._last_start = time.monotonic()
            with self._opener.open(request, timeout=self._timeout) as response:
                body = response.read(max_bytes + 1)
                content_type = response.headers.get("Content-Type")
                content_length = response.headers.get("Content-Length")
                transfer_encoding = response.headers.get("Transfer-Encoding")
                content_range = response.headers.get("Content-Range")
                status = getattr(response, "status", None)
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            raise HTTPTransportError(f"Provider HTTP {status}") from None
        except (OSError, ValueError, http.client.HTTPException):
            raise HTTPTransportError("Provider HTTPS request failed") from None
        if len(body) > max_bytes:
            raise HTTPTransportError("Provider response exceeds max_bytes")
        if content_range is not None or status is not None and status != 200:
            raise HTTPTransportError("Provider GET did not return a complete HTTP 200 response")
        if content_length is not None and transfer_encoding is None:
            if not isinstance(content_length, str) or not content_length.strip().isdigit():
                raise HTTPTransportError("Provider Content-Length is invalid")
            if int(content_length) != len(body):
                raise HTTPTransportError("Provider response differs from Content-Length")
        return HTTPResponse(body, datetime.now(UTC).isoformat(), content_type)
