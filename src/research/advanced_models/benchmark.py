"""
Benchmark module for inference latency and memory profiling.
Measures cold load, warm-up, batch latency (1, 10, 100, 1000 URLs),
percentiles (P50, P90, P95, P99), and memory peaks.
"""

from typing import Dict, Any, List
import time
import tracemalloc
import numpy as np
import pandas as pd


def benchmark_model_inference(
    model: Any,
    X_sample: pd.DataFrame,
    n_repetitions: int = 50,
) -> Dict[str, Any]:
    """
    Measures latency percentiles and memory usage on X_sample.
    """
    n_rows = len(X_sample)

    # 1. Warm-up (not timed)
    for _ in range(5):
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(X_sample)
        elif hasattr(model, "predict"):
            _ = model.predict(X_sample)

    # 2. Timing
    latencies = []
    tracemalloc.start()
    for _ in range(n_repetitions):
        t0 = time.perf_counter()
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(X_sample)
        elif hasattr(model, "predict"):
            _ = model.predict(X_sample)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies = np.array(latencies)
    lat_per_url = latencies / max(1, n_rows)

    return {
        "n_samples": int(n_rows),
        "n_repetitions": int(n_repetitions),
        "mean_total_ms": float(np.mean(latencies)),
        "p50_total_ms": float(np.percentile(latencies, 50)),
        "p90_total_ms": float(np.percentile(latencies, 90)),
        "p95_total_ms": float(np.percentile(latencies, 95)),
        "p99_total_ms": float(np.percentile(latencies, 99)),
        "mean_per_url_ms": float(np.mean(lat_per_url)),
        "p50_per_url_ms": float(np.percentile(lat_per_url, 50)),
        "peak_memory_kb": float(peak_mem / 1024.0),
    }
