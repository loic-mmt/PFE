"""Bounded LSE row reads for candles, corporate actions and selected option chains.

Paths and parameters follow lse-data 0.14's ``client.py``. The SDK's JSON reader
has no byte cap, so these explicit GETs cap bytes before decoding instead. No
bulk export, credentials in query strings, or redirect is permitted.
"""

import http.client
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping

from pfe.data.qualification import (
    VAULT_URL,
    BoundedLSETransport,
    DiscoveryError,
    DiscoveryLimitError,
    DiscoveryResponse,
    _validate_lse_http_body,
    sanitize_metadata,
)

_COMMON = frozenset(("symbol", "start", "end", "order", "limit"))
_PARAMETERS = {
    "/candles": _COMMON | {"timeframe", "dataset"},
    "/ref/stock_splits": _COMMON,
    "/ref/dividends": _COMMON,
    "/options/chain": frozenset(
        ("underlying", "type", "expiry", "strike_min", "strike_max", "limit")
    ),
}


class LSEDataTransport(BoundedLSETransport):
    """GET a capped row page using the existing safe credential/redirect setup."""

    def get(self, endpoint, params: Mapping, *, max_bytes: int) -> DiscoveryResponse:
        if endpoint not in _PARAMETERS:
            raise DiscoveryError("Only allowlisted LSE candle/action/chain endpoints are allowed")
        if not isinstance(params, Mapping) or set(params) - _PARAMETERS[endpoint]:
            raise DiscoveryError("Unsupported LSE row query parameters")
        if type(params.get("limit")) is not int or not 1 <= params["limit"] <= 5000:
            raise DiscoveryError("LSE row requests require an explicit limit between 1 and 5000")
        if type(max_bytes) is not int or max_bytes < 1:
            raise DiscoveryLimitError("No LSE row byte budget remains")
        allowed_types = (str, int, float) if endpoint == "/options/chain" else (str, int)
        if any(
            type(value) not in allowed_types
            or isinstance(value, float)
            and not math.isfinite(value)
            for value in params.values()
            if value is not None
        ):
            raise DiscoveryError("LSE query parameters must be strings, finite numbers or null")
        if any(self._api_key in value for value in params.values() if isinstance(value, str)):
            raise DiscoveryError("Credentials must not appear in LSE query parameters")
        query = urllib.parse.urlencode(
            [(key, value) for key, value in params.items() if value is not None]
        )
        request = urllib.request.Request(
            VAULT_URL + endpoint + "?" + query,
            headers={"x-api-key": self._api_key, "User-Agent": "pfe-lse-data/1"},
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                data = response.read(max_bytes + 1)
                status = getattr(response, "status", None)
                headers = getattr(response, "headers", None)
        except urllib.error.HTTPError as error:
            error.close()
            raise DiscoveryError(f"LSE row HTTP {error.code}") from None
        except (OSError, ValueError, http.client.HTTPException):
            raise DiscoveryError("LSE row request failed") from None
        if len(data) > max_bytes:
            raise DiscoveryLimitError("LSE row response exceeds max_bytes")
        _validate_lse_http_body(data, status=status, headers=headers)
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError):
            raise DiscoveryError("Invalid LSE row JSON") from None
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            raise DiscoveryError("LSE row response must be a JSON array of objects")
        if len(payload) > params["limit"]:
            raise DiscoveryLimitError("LSE row response exceeds the requested row limit")
        return DiscoveryResponse(
            sanitize_metadata(payload, secrets=(self._api_key,)), byte_count=len(data)
        )
