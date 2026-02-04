from .static import StaticProvider

# Auto-register PydanticAIProvider if pydantic-ai is installed.
try:
    from .pydantic_ai_provider import PydanticAIProvider  # noqa: F401
except ImportError:
    PydanticAIProvider = None  # type: ignore[assignment,misc]

__all__ = ["PydanticAIProvider", "StaticProvider"]
