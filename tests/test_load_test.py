import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serving.load_test import summarize_results


def test_summarize_results_computes_percentiles_correctly():
    # 100 latencies, 1ms to 100ms, all successful.
    results = [(float(i), 200) for i in range(1, 101)]
    summary = summarize_results(results, total_wall_seconds=2.0)

    assert summary["n_requests"] == 100
    assert summary["throughput_rps"] == 50.0
    assert summary["error_count"] == 0
    assert summary["error_rate"] == 0.0
    assert summary["latency_ms_p50"] == 50.5  # median of 1..100
    assert summary["latency_ms_p95"] == 96.0  # index 95 (0-indexed) of sorted 1..100
    assert summary["latency_ms_max"] == 100.0


def test_summarize_results_counts_errors():
    results = [(10.0, 200), (12.0, 200), (15.0, 500), (20.0, 401)]
    summary = summarize_results(results, total_wall_seconds=1.0)

    assert summary["error_count"] == 2
    assert summary["error_rate"] == 0.5
