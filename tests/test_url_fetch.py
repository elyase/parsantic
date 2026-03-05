from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from parsantic.extract.types import Document


def test_from_url_creates_document_with_fetched_text():
    url = "https://example.com/article"
    response = Mock(text="fetched content")
    response.raise_for_status = Mock()
    mock_get = Mock(return_value=response)
    fake_httpx = SimpleNamespace(get=mock_get)

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        doc = Document.from_url(url, additional_context="source=web")

    assert doc.text == "fetched content"
    assert doc.additional_context == "source=web"


def test_from_url_sets_document_id_to_url_when_not_provided():
    url = "https://example.com/default-id"
    response = Mock(text="ok")
    response.raise_for_status = Mock()
    mock_get = Mock(return_value=response)
    fake_httpx = SimpleNamespace(get=mock_get)

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        doc = Document.from_url(url)

    assert doc.document_id == url


def test_from_url_uses_custom_document_id():
    response = Mock(text="ok")
    response.raise_for_status = Mock()
    mock_get = Mock(return_value=response)
    fake_httpx = SimpleNamespace(get=mock_get)

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        doc = Document.from_url("https://example.com/custom-id", document_id="doc-123")

    assert doc.document_id == "doc-123"


def test_from_url_raises_import_error_when_httpx_missing():
    with patch.dict(sys.modules, {"httpx": None}):
        with pytest.raises(
            ImportError,
            match="httpx is required for URL fetching. Install with: pip install parsantic\\[web\\]",
        ):
            Document.from_url("https://example.com/missing-httpx")


def test_afrom_url_with_mocked_async_client():
    url = "https://example.com/async"
    response = Mock(text="async content")
    response.raise_for_status = Mock()
    mock_get = AsyncMock(return_value=response)

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.get = mock_get

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    async_client_ctor = Mock(return_value=FakeAsyncClient())
    fake_httpx = SimpleNamespace(AsyncClient=async_client_ctor)
    headers = {"User-Agent": "parsantic-tests"}

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        doc = asyncio.run(
            Document.afrom_url(url, timeout=12.5, headers=headers, additional_context="mode=async")
        )

    assert doc.text == "async content"
    assert doc.document_id == url
    assert doc.additional_context == "mode=async"
    async_client_ctor.assert_called_once_with()
    mock_get.assert_awaited_once_with(
        url,
        timeout=12.5,
        headers=headers,
        follow_redirects=True,
    )


def test_is_url_helper():
    assert Document._is_url("http://example.com")
    assert Document._is_url("https://example.com")
    assert not Document._is_url("ftp://example.com")
    assert not Document._is_url("example.com")


def test_from_url_http_error_raises_httpx_httpstatuserror():
    class FakeHTTPStatusError(Exception):
        pass

    response = Mock(text="")
    response.raise_for_status = Mock(side_effect=FakeHTTPStatusError("bad status"))
    mock_get = Mock(return_value=response)
    fake_httpx = SimpleNamespace(get=mock_get, HTTPStatusError=FakeHTTPStatusError)

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        with pytest.raises(fake_httpx.HTTPStatusError):
            Document.from_url("https://example.com/error")


def test_from_url_passes_custom_headers_and_timeout():
    url = "https://example.com/config"
    headers = {"Authorization": "Bearer test-token"}
    timeout = 5.25
    response = Mock(text="configured")
    response.raise_for_status = Mock()
    mock_get = Mock(return_value=response)
    fake_httpx = SimpleNamespace(get=mock_get)

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        doc = Document.from_url(url, headers=headers, timeout=timeout)

    assert doc.text == "configured"
    mock_get.assert_called_once_with(
        url,
        timeout=timeout,
        headers=headers,
        follow_redirects=True,
    )
