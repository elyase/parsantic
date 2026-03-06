from .static import StaticProvider

# Auto-register PydanticAIProvider if pydantic-ai is installed.
try:
    from .pydantic_ai_provider import PydanticAIProvider  # noqa: F401

    __all__ = ["PydanticAIProvider", "StaticProvider"]
except ImportError:
    __all__ = ["StaticProvider"]
