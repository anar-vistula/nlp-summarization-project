"""
Cost and resource accounting for experiment runs.

Every model run is priced on three axes so the thesis can compare systems on
effectiveness *and* on what they cost to obtain:

  - API cost: real token usage reported by the provider, priced per model.
  - Compute time: wall-clock and process CPU time measured locally.
  - Notional compute cost: time converted to money at a configurable hourly
    rate, so a "free" local model and a paid API can be put on one axis.

The prices below are rates as of the date in PRICES_AS_OF and WILL go stale.
They are data, not truth: check them against the provider's current pricing
page before quoting any figure in the thesis. Where a run reports real token
usage, the token counts themselves are stored alongside the derived cost, so
every number can be re-priced later without re-running anything.
"""

import os
import platform
import resource
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

PRICES_AS_OF = "2026-09-27"

# USD per 1 million tokens, keyed by the provider's model id.
API_PRICES: Dict[str, Dict[str, float]] = {
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}

# Notional hourly rates for locally spent time. The local machine is already
# paid for, so its "cost" is an opportunity cost; the Colab figure is what an
# equivalent cloud T4 would bill. Both are reported separately from real API
# spending and are clearly labelled as notional in the report.
COMPUTE_RATES_USD_PER_HOUR = {
    "local_cpu": float(os.environ.get("RATE_LOCAL_CPU_USD_HOUR", 0.10)),
    "colab_gpu": float(os.environ.get("RATE_COLAB_GPU_USD_HOUR", 0.35)),
}

# Which rate applies to which model type (see model_registry.MODEL_TYPES).
TYPE_TO_RATE = {
    "extractive": "local_cpu",
    "hf": "local_cpu",
    "colab": "colab_gpu",
    "paid": None,  # priced from real token usage instead
}


@dataclass
class ResourceUsage:
    """What one model run consumed."""

    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0          # user+system CPU of this process
    gpu_seconds: Optional[float] = None   # reported by the Colab server
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    api_cost_usd: float = 0.0         # real money, from provider token counts
    compute_cost_usd: float = 0.0     # notional, from time x hourly rate
    cost_basis: str = "none"          # "api_usage" | "notional_time" | "none"
    priced_model: Optional[str] = None

    @property
    def total_cost_usd(self) -> float:
        return round(self.api_cost_usd + self.compute_cost_usd, 6)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["total_cost_usd"] = self.total_cost_usd
        return data


class UsageTimer:
    """
    Context manager measuring wall-clock and process CPU time around a run.

    CPU time is taken from getrusage, so it captures the work actually done by
    this process (a CPU-bound transformer shows cpu ≈ wall × cores, while a
    network call to Colab shows cpu ≈ 0). That difference is what separates
    local compute from remote compute in the report.
    """

    def __init__(self):
        self.wall_seconds = 0.0
        self.cpu_seconds = 0.0

    def __enter__(self):
        self._wall0 = time.perf_counter()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self._cpu0 = usage.ru_utime + usage.ru_stime
        return self

    def __exit__(self, *exc):
        self.wall_seconds = time.perf_counter() - self._wall0
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu_seconds = (usage.ru_utime + usage.ru_stime) - self._cpu0
        return False


def price_api_call(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """USD for a provider call, from real token counts. 0.0 if unpriced."""
    price = API_PRICES.get(model_id)
    if not price:
        return 0.0
    return round(
        (input_tokens / 1_000_000) * price["input"]
        + (output_tokens / 1_000_000) * price["output"],
        6,
    )


def estimate_api_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Same pricing, used before a run to forecast spend (see --dry-run)."""
    return price_api_call(model_id, input_tokens, output_tokens)


def notional_compute_cost(model_type: str, wall_seconds: float) -> float:
    """Time on local CPU or Colab GPU converted to money at the configured rate."""
    rate_key = TYPE_TO_RATE.get(model_type)
    if not rate_key:
        return 0.0
    return round(COMPUTE_RATES_USD_PER_HOUR[rate_key] * (wall_seconds / 3600.0), 6)


def build_usage(
    model_type: str,
    timer: UsageTimer,
    usage_payload: Optional[Dict] = None,
    model_id: Optional[str] = None,
    gpu_seconds: Optional[float] = None,
) -> ResourceUsage:
    """
    Assemble a ResourceUsage from a finished timer plus optional API usage.

    For Colab models the wall clock includes the network round-trip, which is
    not GPU time. When the server reports its own gpu_seconds, that is what
    gets priced, and the wall clock is still recorded for reference.
    """
    entry = ResourceUsage(
        wall_seconds=round(timer.wall_seconds, 4),
        cpu_seconds=round(timer.cpu_seconds, 4),
        gpu_seconds=round(gpu_seconds, 4) if gpu_seconds is not None else None,
    )
    if usage_payload and model_id:
        entry.input_tokens = usage_payload.get("input_tokens")
        entry.output_tokens = usage_payload.get("output_tokens")
        entry.api_cost_usd = price_api_call(
            model_id, entry.input_tokens or 0, entry.output_tokens or 0
        )
        entry.cost_basis = "api_usage"
        entry.priced_model = model_id
        if model_id not in API_PRICES:
            entry.cost_basis = "api_usage_unpriced"
    else:
        billable = gpu_seconds if (model_type == "colab" and gpu_seconds) else timer.wall_seconds
        entry.compute_cost_usd = notional_compute_cost(model_type, billable)
        entry.cost_basis = "notional_time" if entry.compute_cost_usd else "none"
    return entry


def machine_fingerprint() -> Dict:
    """Recorded in every experiment manifest: timings only mean something with it."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "prices_as_of": PRICES_AS_OF,
        "compute_rates_usd_per_hour": dict(COMPUTE_RATES_USD_PER_HOUR),
    }
