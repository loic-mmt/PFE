"""LSE metadata and row collectors must never accept incomplete HTTP evidence."""

import http.client
import io
import urllib.error

import pytest

from pfe.data.lse_transport import LSEDataTransport
from pfe.data.qualification import BoundedLSETransport, DiscoveryError, DiscoveryLimitError


class Response(io.BytesIO):
    def __init__(self, *, status=200, headers=None):
        super().__init__(b"[]")
        self.status = status
        self.headers = {"Content-Length": "2"} if headers is None else headers


@pytest.fixture(params=["metadata", "row"])
def request_lse(request):
    def fetch(opener, *, max_bytes=100):
        if request.param == "metadata":
            return BoundedLSETransport("synthetic-secret", opener=opener).get(
                "/catalog", max_bytes=max_bytes
            )
        return LSEDataTransport("synthetic-secret", opener=opener).get(
            "/candles", {"symbol": "AAPL", "limit": 10}, max_bytes=max_bytes
        )

    return fetch


def opener_for(response):
    class Opener:
        def open(self, request, *, timeout):
            return response

    return Opener()


@pytest.mark.parametrize("status", [201, 206, 302, 500])
def test_non_200_response_is_rejected_and_closed(request_lse, status):
    response = Response(status=status)
    with pytest.raises(DiscoveryError, match="complete HTTP 200"):
        request_lse(opener_for(response))
    assert response.closed


def test_unsolicited_range_is_rejected_even_with_200_status(request_lse):
    response = Response(headers={"Content-Range": "bytes 0-1/1000"})
    with pytest.raises(DiscoveryError, match="complete HTTP 200"):
        request_lse(opener_for(response))
    assert response.closed


@pytest.mark.parametrize("length", ["1", "1000", "bad", "-1", "9" * 5000])
def test_truncated_or_invalid_length_is_rejected_and_closed(request_lse, length):
    response = Response(headers={"Content-Length": length})
    with pytest.raises(DiscoveryError, match="Content-Length"):
        request_lse(opener_for(response))
    assert response.closed


def test_complete_response_and_legacy_fakes_remain_supported(request_lse):
    for response in (Response(), io.BytesIO(b"[]")):
        assert request_lse(opener_for(response)).payload == []
        assert response.closed


def test_byte_cap_precedes_decode_or_length_validation(request_lse):
    response = Response(headers={"Content-Length": "1000"})
    with pytest.raises(DiscoveryLimitError, match="max_bytes"):
        request_lse(opener_for(response), max_bytes=1)
    assert response.closed


def test_incomplete_read_does_not_leak_partial_body_or_credentials(request_lse):
    class BrokenResponse(Response):
        def read(self, count):
            raise http.client.IncompleteRead(b"synthetic-secret", 100)

    response = BrokenResponse()
    with pytest.raises(DiscoveryError, match="request failed") as failure:
        request_lse(opener_for(response))
    assert response.closed
    assert "synthetic-secret" not in str(failure.value)


def test_http_error_body_is_closed_and_not_exposed(request_lse):
    response = io.BytesIO(b"synthetic-secret")

    class Opener:
        def open(self, request, *, timeout):
            raise urllib.error.HTTPError(request.full_url, 403, "synthetic-secret", {}, response)

    with pytest.raises(DiscoveryError, match="HTTP 403") as failure:
        request_lse(Opener())
    assert response.closed
    assert "synthetic-secret" not in str(failure.value)
