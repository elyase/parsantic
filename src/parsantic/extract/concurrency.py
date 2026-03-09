from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ConcurrencyConfig:
    network_workers: int = 4
    cpu_workers: int = 2
    max_inflight_image_bytes: int = 100_000_000

    def __post_init__(self) -> None:
        if self.network_workers < 1:
            raise ValueError("concurrency.network_workers must be >= 1")
        if self.cpu_workers < 1:
            raise ValueError("concurrency.cpu_workers must be >= 1")
        if self.max_inflight_image_bytes < 1:
            raise ValueError("concurrency.max_inflight_image_bytes must be >= 1")


@dataclass(slots=True)
class BudgetTracker:
    max_api_calls: int | None = None
    max_inflight_image_bytes: int | None = None
    api_calls: int = 0
    degraded_to_text_only: bool = False
    repeated_server_errors: int = 0

    def consume_api_calls(self, count: int) -> None:
        self.api_calls += count
        if self.max_api_calls is not None and self.api_calls > self.max_api_calls:
            raise ValueError(f"LLM call budget exceeded: {self.api_calls} > {self.max_api_calls}")

    def ensure_image_bytes(self, count: int) -> None:
        if self.max_inflight_image_bytes is None:
            return
        if count > self.max_inflight_image_bytes:
            raise ValueError(
                "Rasterized image bytes exceed concurrency.max_inflight_image_bytes: "
                f"{count} > {self.max_inflight_image_bytes}"
            )

    def on_http_429(self, network_workers: int) -> int:
        self.degraded_to_text_only = False
        return max(1, network_workers // 2)

    def on_server_error(self) -> bool:
        self.repeated_server_errors += 1
        self.degraded_to_text_only = self.repeated_server_errors >= 2
        return self.degraded_to_text_only
