"""Tests for Vertex AI provider support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from parsantic.extract.providers.factory import _kwargs_with_environment_defaults
from parsantic.extract.providers.pydantic_ai_provider import (
    _build_model_with_credentials,
    _parse_model_spec,
)


class TestParseModelSpec:
    def test_vertex_model_spec(self):
        provider, model = _parse_model_spec("vertex:gemini-2.5-flash")
        assert provider == "vertex"
        assert model == "gemini-2.5-flash"

    def test_vertex_model_with_version(self):
        provider, model = _parse_model_spec("vertex:gemini-2.0-flash-001")
        assert provider == "vertex"
        assert model == "gemini-2.0-flash-001"


class TestPydanticAIProviderVertexFields:
    def test_post_init_passes_vertex_kwargs(self):
        """Verify __post_init__ passes vertex fields to _build_model_with_credentials."""
        from parsantic.extract.providers.pydantic_ai_provider import PydanticAIProvider

        mock_model = MagicMock()
        with (
            patch(
                "parsantic.extract.providers.pydantic_ai_provider._build_model_with_credentials",
                return_value=mock_model,
            ) as mock_build,
            patch("parsantic.extract.providers.pydantic_ai_provider.Agent"),
        ):
            PydanticAIProvider(
                model_id="vertex:gemini-2.5-flash",
                project_id="my-project",
                region="europe-west1",
                service_account_file="/path/to/sa.json",
            )

        mock_build.assert_called_once_with(
            "vertex:gemini-2.5-flash",
            api_key=None,
            base_url=None,
            project_id="my-project",
            region="europe-west1",
            service_account_file="/path/to/sa.json",
        )


class TestBuildModelVertex:
    def test_vertex_creates_google_model_with_google_provider(self):
        """Vertex branch uses GoogleProvider(vertexai=True) when available."""
        mock_google_model_cls = MagicMock()
        mock_google_provider_cls = MagicMock()
        mock_provider_instance = MagicMock()
        mock_google_provider_cls.return_value = mock_provider_instance

        mock_google_module = MagicMock(GoogleModel=mock_google_model_cls)
        mock_provider_module = MagicMock(GoogleProvider=mock_google_provider_cls)

        with patch.dict(
            "sys.modules",
            {
                "pydantic_ai.models.google": mock_google_module,
                "pydantic_ai.providers.google": mock_provider_module,
            },
        ):
            result = _build_model_with_credentials(
                "vertex:gemini-2.5-flash",
                project_id="my-project",
                region="us-central1",
            )

        mock_google_provider_cls.assert_called_once_with(
            vertexai=True, project="my-project", location="us-central1"
        )
        mock_google_model_cls.assert_called_once_with(
            "gemini-2.5-flash", provider=mock_provider_instance
        )
        assert result == mock_google_model_cls.return_value

    def test_vertex_with_no_extra_kwargs(self):
        """Vertex branch with no project/region passes only vertexai=True."""
        mock_google_model_cls = MagicMock()
        mock_google_provider_cls = MagicMock()

        mock_google_module = MagicMock(GoogleModel=mock_google_model_cls)
        mock_provider_module = MagicMock(GoogleProvider=mock_google_provider_cls)

        with patch.dict(
            "sys.modules",
            {
                "pydantic_ai.models.google": mock_google_module,
                "pydantic_ai.providers.google": mock_provider_module,
            },
        ):
            _build_model_with_credentials("vertex:gemini-2.5-flash")

        mock_google_provider_cls.assert_called_once_with(vertexai=True)

    def test_vertex_service_account_file_modern_path(self):
        """service_account_file loads credentials via google.auth for GoogleProvider."""
        mock_google_model_cls = MagicMock()
        mock_google_provider_cls = MagicMock()
        mock_creds = MagicMock()

        mock_google_module = MagicMock(GoogleModel=mock_google_model_cls)
        mock_provider_module = MagicMock(GoogleProvider=mock_google_provider_cls)

        with (
            patch.dict(
                "sys.modules",
                {
                    "pydantic_ai.models.google": mock_google_module,
                    "pydantic_ai.providers.google": mock_provider_module,
                },
            ),
            patch(
                "google.auth.load_credentials_from_file",
                return_value=(mock_creds, "project"),
            ) as mock_load,
        ):
            _build_model_with_credentials(
                "vertex:gemini-2.5-flash",
                service_account_file="/path/to/sa.json",
            )

        mock_load.assert_called_once_with("/path/to/sa.json")
        mock_google_provider_cls.assert_called_once_with(vertexai=True, credentials=mock_creds)

    def test_vertex_bad_service_account_file_propagates_error(self):
        """Invalid service_account_file raises, not silently swallowed."""
        mock_google_model_cls = MagicMock()
        mock_google_provider_cls = MagicMock()

        mock_google_module = MagicMock(GoogleModel=mock_google_model_cls)
        mock_provider_module = MagicMock(GoogleProvider=mock_google_provider_cls)

        with (
            patch.dict(
                "sys.modules",
                {
                    "pydantic_ai.models.google": mock_google_module,
                    "pydantic_ai.providers.google": mock_provider_module,
                },
            ),
            patch(
                "google.auth.load_credentials_from_file",
                side_effect=FileNotFoundError("no such file"),
            ),
            pytest.raises(FileNotFoundError, match="no such file"),
        ):
            _build_model_with_credentials(
                "vertex:gemini-2.5-flash",
                service_account_file="/bad/path.json",
            )

    def test_vertex_falls_back_to_legacy_provider(self):
        """Falls back to GoogleVertexProvider when GoogleProvider import fails."""
        mock_google_model_cls = MagicMock()
        mock_vertex_provider_cls = MagicMock()
        mock_vertex_instance = MagicMock()
        mock_vertex_provider_cls.return_value = mock_vertex_instance

        mock_google_module = MagicMock(GoogleModel=mock_google_model_cls)
        mock_vertex_module = MagicMock(GoogleVertexProvider=mock_vertex_provider_cls)

        with patch.dict(
            "sys.modules",
            {
                "pydantic_ai.models.google": mock_google_module,
                "pydantic_ai.providers.google": None,  # Force ImportError
                "pydantic_ai.providers.google_vertex": mock_vertex_module,
            },
        ):
            result = _build_model_with_credentials(
                "vertex:gemini-2.5-flash",
                project_id="my-project",
                region="us-central1",
            )

        mock_vertex_provider_cls.assert_called_once_with(
            project_id="my-project", region="us-central1"
        )
        mock_google_model_cls.assert_called_once_with(
            "gemini-2.5-flash", provider=mock_vertex_instance
        )
        assert result == mock_google_model_cls.return_value

    def test_vertex_service_account_file_with_legacy_fallback(self):
        """service_account_file is passed to legacy GoogleVertexProvider."""
        mock_google_model_cls = MagicMock()
        mock_vertex_provider_cls = MagicMock()

        mock_google_module = MagicMock(GoogleModel=mock_google_model_cls)
        mock_vertex_module = MagicMock(GoogleVertexProvider=mock_vertex_provider_cls)

        with patch.dict(
            "sys.modules",
            {
                "pydantic_ai.models.google": mock_google_module,
                "pydantic_ai.providers.google": None,
                "pydantic_ai.providers.google_vertex": mock_vertex_module,
            },
        ):
            _build_model_with_credentials(
                "vertex:gemini-2.5-flash",
                service_account_file="/path/to/sa.json",
            )

        mock_vertex_provider_cls.assert_called_once_with(service_account_file="/path/to/sa.json")

    def test_vertex_returns_fallback_string_on_total_import_failure(self):
        """Returns model_spec string when all google imports fail."""
        with patch.dict(
            "sys.modules",
            {
                "pydantic_ai.models.google": None,
                "pydantic_ai.providers.google": None,
                "pydantic_ai.providers.google_vertex": None,
            },
        ):
            result = _build_model_with_credentials(
                "vertex:gemini-2.5-flash",
                project_id="my-project",
            )
        assert result == "vertex:gemini-2.5-flash"


class TestFactoryVertexEnvVars:
    def test_resolves_vertex_project_id(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VERTEX_PROJECT_ID", "env-project")
        resolved = _kwargs_with_environment_defaults("vertex:gemini-2.5-flash", {})
        assert resolved["project_id"] == "env-project"

    def test_resolves_vertex_region(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VERTEX_REGION", "europe-west1")
        resolved = _kwargs_with_environment_defaults("vertex:gemini-2.5-flash", {})
        assert resolved["region"] == "europe-west1"

    def test_resolves_google_application_credentials(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")
        resolved = _kwargs_with_environment_defaults("vertex:gemini-2.5-flash", {})
        assert resolved["service_account_file"] == "/path/to/sa.json"

    def test_explicit_kwargs_override_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VERTEX_PROJECT_ID", "env-project")
        monkeypatch.setenv("VERTEX_REGION", "env-region")
        resolved = _kwargs_with_environment_defaults(
            "vertex:gemini-2.5-flash",
            {"project_id": "explicit-project", "region": "explicit-region"},
        )
        assert resolved["project_id"] == "explicit-project"
        assert resolved["region"] == "explicit-region"

    def test_no_vertex_env_vars_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
        monkeypatch.delenv("VERTEX_REGION", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        resolved = _kwargs_with_environment_defaults("vertex:gemini-2.5-flash", {})
        assert "project_id" not in resolved
        assert "region" not in resolved
        assert "service_account_file" not in resolved
