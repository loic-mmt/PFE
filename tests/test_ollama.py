import io
import json

import pytest

from pfe import OllamaBackend

DIGEST = "1" * 64


class Response(io.BytesIO):
    status = 200

    def __init__(self, payload, *, length=None):
        body = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body) if length is None else length)}


class Opener:
    def __init__(self, response=None, digest=DIGEST, context=8192, remote=False):
        self.calls, self.response, self.digest = [], response, digest
        self.context, self.remote = context, remote

    def open(self, request, **kwargs):
        self.calls.append(request)
        if request.full_url.endswith("/api/tags"):
            return Response({"models": [{"name": "local:latest", "digest": self.digest}]})
        if request.full_url.endswith("/api/show"):
            return Response(
                {
                    "model_info": {"llama.context_length": self.context},
                    "details": {"format": "gguf"},
                    "remote_host": "cloud" if self.remote else None,
                }
            )
        return Response(self.response)


def test_chat_only_installed_digest_structured_nonstreaming_no_pull():
    opener = Opener({"message": {"content": '{"events":[]}'}, "done": True, "done_reason": "stop"})
    backend = OllamaBackend("local", model_revision=DIGEST, opener=opener)
    response = backend.chat([{"role": "user", "content": "source"}], {"type": "object"})
    assert response.complete and response.content == '{"events":[]}'
    calls = [request.full_url.rsplit("/", 1)[-1] for request in opener.calls]
    assert calls == ["tags", "show", "chat", "tags"]
    payload = json.loads(opener.calls[2].data)
    assert payload["stream"] is False and payload["keep_alive"] == 0
    assert "tools" not in payload and payload["options"]["temperature"] == 0


def test_embeddings_do_not_silently_truncate_or_download():
    opener = Opener({"embeddings": [[0.1, 0.2]]})
    response = OllamaBackend("local", model_revision=DIGEST, opener=opener).embed("small source")
    assert json.loads(response.content) == {"embedding": [0.1, 0.2]}
    assert json.loads(opener.calls[2].data)["truncate"] is False


def test_missing_changed_digest_aborts_before_inference():
    opener = Opener(digest="2" * 64)
    with pytest.raises(ValueError, match="digest"):
        OllamaBackend("local", model_revision=DIGEST, opener=opener).chat([], {})
    assert len(opener.calls) == 1


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1",
        "http://example.org",
        "http://127.0.0.1/path",
        "http://user:pass@localhost",
    ],
)
def test_backend_requires_loopback(endpoint):
    with pytest.raises(ValueError):
        OllamaBackend("local", model_revision=DIGEST, endpoint=endpoint)


def test_truncated_model_output_and_capped_transport():
    opener = Opener({"message": {"content": "{}"}, "done": True, "done_reason": "length"})
    assert not OllamaBackend("local", model_revision=DIGEST, opener=opener).chat([], {}).complete
    with pytest.raises(ValueError, match="byte limit"):
        OllamaBackend("local", model_revision=DIGEST, opener=opener, max_response_bytes=2).chat(
            [], {}
        )


def test_context_cap_and_cloud_alias_preflight_no_inference():
    opener = Opener(context=4096)
    with pytest.raises(ValueError, match="context budget"):
        OllamaBackend("local", model_revision=DIGEST, opener=opener).chat(
            [{"role": "user", "content": "x" * 4096}], {}
        )
    assert [request.full_url.rsplit("/", 1)[-1] for request in opener.calls] == ["tags", "show"]
    opener = Opener(remote=True)
    with pytest.raises(ValueError, match="cloud aliases"):
        OllamaBackend("local", model_revision=DIGEST, opener=opener).embed("source")
    assert len(opener.calls) == 2
