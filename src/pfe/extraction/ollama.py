"""Opt-in loopback Ollama API; only installed models, no pull/tool execution."""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit

from pfe.data.contracts import _instant
from pfe.data.documents import utc_now
from pfe.data.http import _NoRedirect
from pfe.data.storage import _json_bytes
from pfe.extraction.validation import strict_json


@dataclass(frozen=True)
class ModelResponse:
    content: str
    received_at: str
    complete: bool = True

    def __post_init__(self):
        if not isinstance(self.content, str) or type(self.complete) is not bool:
            raise ValueError("ModelResponse requires text content and boolean complete")
        _instant(self.received_at, "received_at")


class OllamaBackend:
    """Small non-streaming calls against an already installed, digest-pinned model.

    Aliases are checked through /api/tags before and after inference. Changing the
    installed model invalidates the run rather than reusing an alias-based cache.
    Local HTTP has no proxy, redirect, automatic retry or automatic model pull.
    """

    def __init__(
        self,
        model,
        *,
        model_revision,
        endpoint="http://127.0.0.1:11434",
        options=None,
        timeout=60,
        max_response_bytes=65_536,
        opener=None,
    ):
        parts = urlsplit(endpoint)
        if (
            parts.scheme != "http"
            or parts.hostname not in ("127.0.0.1", "localhost", "::1")
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
            or parts.username is not None
            or parts.password is not None
        ):
            raise ValueError("Ollama endpoint must be loopback HTTP without credentials/path")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Ollama model must be an already installed model name")
        import re

        if not isinstance(model_revision, str) or not re.fullmatch(r"[0-9a-f]{64}", model_revision):
            raise ValueError(
                "model_revision must be the installed model SHA256 digest from /api/tags"
            )
        if type(timeout) not in (int, float) or not 0 < timeout <= 60:
            raise ValueError("Ollama timeout must be between 0 and 60 seconds")
        if type(max_response_bytes) is not int or max_response_bytes < 1:
            raise ValueError("max_response_bytes must be a positive integer")
        self.model, self.model_revision = model, model_revision
        self.endpoint = endpoint.rstrip("/")
        self.options = {"temperature": 0, "num_predict": 2048, "num_ctx": 32768, **(options or {})}
        for name, maximum in (("num_predict", 4096), ("num_ctx", 32768)):
            if type(self.options[name]) is not int or not 1 <= self.options[name] <= maximum:
                raise ValueError(f"Ollama {name} must be a bounded positive integer <= {maximum}")
        _json_bytes(self.options)
        self.timeout, self.max_response_bytes = timeout, max_response_bytes
        self._opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )

    @property
    def identity(self):
        return {
            "backend": "ollama-loopback-1",
            "model": self.model,
            "model_revision": self.model_revision,
            "options": self.options,
        }

    def _request(self, path, payload=None):
        request = urllib.request.Request(
            self.endpoint + path,
            data=None if payload is None else _json_bytes(payload),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise ValueError("Ollama response must be HTTP 200")
                length = response.headers.get("Content-Length")
                if length is not None and (
                    not length.isdigit() or int(length) > self.max_response_bytes
                ):
                    raise ValueError("Ollama response byte limit exceeded or invalid length")
                body = response.read(self.max_response_bytes + 1)
                if len(body) > self.max_response_bytes or (
                    length is not None and len(body) != int(length)
                ):
                    raise ValueError("Ollama response is oversized or incomplete")
        except (OSError, urllib.error.URLError):
            raise ValueError(
                "Local Ollama request failed; install/start the model separately"
            ) from None
        return strict_json(body.decode("utf-8")), utc_now()

    def _check_digest(self):
        payload, _ = self._request("/api/tags")
        target = self.model if ":" in self.model else self.model + ":latest"
        matches = [
            entry
            for entry in payload.get("models", [])
            if entry.get("name") in (self.model, target)
            or entry.get("model") in (self.model, target)
        ]
        if len(matches) != 1 or matches[0].get("digest") != self.model_revision:
            raise ValueError("Ollama model missing or installed digest differs from model_revision")

    def _local_details(self):
        details, _ = self._request("/api/show", {"model": self.model, "verbose": False})
        if (
            details.get("details", {}).get("format") != "gguf"
            or details.get("remote_host")
            or details.get("remote_model")
            or self.model.endswith("-cloud")
        ):
            raise ValueError(
                "Ollama extraction requires an installed local GGUF model; "
                "cloud aliases are refused"
            )
        return details

    def chat(self, messages, schema):
        self._check_digest()
        details = self._local_details()
        contexts = [
            value
            for key, value in details.get("model_info", {}).items()
            if key.endswith(".context_length") and type(value) is int and value > 0
        ]
        if not contexts:
            raise ValueError("Ollama model context length is unknown; inference was not started")
        context = min(self.options["num_ctx"], min(contexts))
        text_bytes = sum(len(message["content"].encode("utf-8")) for message in messages)
        template_bytes = len(details.get("template", "").encode("utf-8"))
        system_bytes = len(details.get("system", "").encode("utf-8"))
        # Conservative UTF-8 byte bound with output/template reserve, rather than
        # average chars/token. Real template/model recall still needs corpus audit.
        if text_bytes + template_bytes + system_bytes + 512 + self.options["num_predict"] > context:
            raise ValueError(
                "Ollama input exceeds conservative context budget; reduce chunk_chars/num_predict"
            )
        payload, received = self._request(
            "/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "format": schema,
                "options": {**self.options, "num_ctx": context},
                "stream": False,
                "keep_alive": 0,
            },
        )
        self._check_digest()
        content = payload.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama chat response lacks text content")
        return ModelResponse(
            content, received, payload.get("done") is True and payload.get("done_reason") == "stop"
        )

    def embed(self, text):
        self._check_digest()
        self._local_details()
        payload, received = self._request(
            "/api/embed",
            {
                "model": self.model,
                "input": text,
                "truncate": False,
                "keep_alive": 0,
            },
        )
        self._check_digest()
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != 1:
            raise ValueError("Ollama embed requires exactly one vector")
        return ModelResponse(json.dumps({"embedding": vectors[0]}), received)
