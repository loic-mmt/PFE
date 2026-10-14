"""Capped HTTPS reads, provider errors, credentials and endpoint restrictions."""

import io
import urllib.error

import pytest

from pfe.data.http import BoundedHTTPTransport, HTTPResponse, HTTPTransportError


class Response(io.BytesIO):
    headers = {"Content-Type": "application/json"}


def test_body_cap_is_applied_before_decoding_and_preserves_bytes():
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            return Response(b'{"ok":true}')

    client = BoundedHTTPTransport(opener=Opener())
    response = client.get(
        "https://api.stlouisfed.org/fred/series",
        max_bytes=100,
        params={"api_key": "synthetic-key", "series_id": "CPI"},
    )
    assert response.body == b'{"ok":true}' and response.byte_count == 11
    assert response.content_type == "application/json"
    assert requests[0].get_method() == "GET"
    with pytest.raises(HTTPTransportError, match="exceeds max_bytes"):
        client.get("https://api.stlouisfed.org/fred/series", max_bytes=2)


def test_provider_error_never_exposes_url_query_or_response_body():
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 429, "synthetic-secret", {}, Response(b"synthetic-secret")
            )

    client = BoundedHTTPTransport(opener=Opener())
    with pytest.raises(HTTPTransportError) as error:
        client.get(
            "https://api.eia.gov/v2/foo", params={"api_key": "synthetic-secret"}, max_bytes=100
        )
    assert str(error.value) == "Provider HTTP 429"


@pytest.mark.parametrize(
    "url",
    [
        "http://data.sec.gov/foo",
        "https://unrelated.test/foo",
        "file:///tmp/foo",
        "https://user:pass@data.sec.gov/foo",
        "https://data.sec.gov:444/foo",
        "https://data.sec.gov/foo#fragment",
    ],
)
def test_unsupported_urls_stop_before_network(url):
    class Opener:
        def open(self, *args, **kwargs):
            raise AssertionError("Invalid URL reached provider")

    with pytest.raises(ValueError, match="allowed host"):
        BoundedHTTPTransport(opener=Opener()).get(url, max_bytes=100)


def test_response_receipt_and_headers_are_explicit():
    with pytest.raises(ValueError, match="aware timestamp"):
        HTTPResponse(b"data", "2026-01-01")
    with pytest.raises(ValueError, match="control characters"):
        BoundedHTTPTransport(headers={"User-Agent": "name\r\nAuthorization: secret"})
    with pytest.raises(ValueError, match="finite"):
        BoundedHTTPTransport(timeout=float("nan"))
    with pytest.raises(ValueError, match="positive"):
        BoundedHTTPTransport().get("https://data.sec.gov/foo", max_bytes=True)


@pytest.mark.parametrize("length", ["1000", "2", "-1", "bad"])
def test_truncated_or_invalid_declared_body_never_becomes_a_complete_file(length):
    class Opener:
        def open(self, request, timeout):
            response = Response(b"DATE,CLOSE\n10/12/2026,20\n")
            response.headers = {"Content-Length": length}
            return response

    with pytest.raises(HTTPTransportError, match="Content-Length"):
        BoundedHTTPTransport(opener=Opener()).get(
            "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
            max_bytes=100,
        )


def test_unsolicited_partial_response_never_becomes_a_complete_source_file():
    class Opener:
        def open(self, request, timeout):
            response = Response(b"DATE,CLOSE\n10/12/2026,20\n")
            response.headers = {"Content-Range": "bytes 0-24/1000"}
            response.status = 206
            return response

    with pytest.raises(HTTPTransportError, match="complete HTTP 200"):
        BoundedHTTPTransport(opener=Opener()).get(
            "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
            max_bytes=100,
        )
