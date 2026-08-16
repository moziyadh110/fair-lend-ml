"""Concurrent load test against a running serving/app.py instance — checks
whether the dashboard's/README's latency claims are backed by anything, since
nothing had actually measured this before (see PROGRESS.md).

Uses httpx (already a project dependency, for the async client) rather than
adding locust/k6 as a new dependency for one script.

Usage: uvicorn serving.app:app &   (or docker compose up -d api)
       python serving/load_test.py --requests 500 --concurrency 50
"""
import argparse
import asyncio
import os
import statistics
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

VALID_APPLICATION = {
    "amt_income_total": 150000,
    "amt_credit": 500000,
    "amt_annuity": 25000,
    "amt_goods_price": 450000,
    "age_years": 35,
    "years_employed": 5,
    "owns_car": 1,
    "owns_realty": 1,
    "cnt_children": 0,
    "bureau_loan_count": 2,
    "active_loans": 1,
    "closed_loans": 1,
    "avg_days_overdue": 0,
    "max_days_overdue": 0,
    "bureau_debt_ratio": 0.2,
    "total_bureau_credit": 100000,
    "prev_app_count": 1,
    "prev_approved": 1,
    "prev_refused": 0,
    "credit_application_ratio": 1.0,
    "late_payments": 0,
    "avg_days_late": 0,
    "payment_volatility": 0,
    "total_installments": 12,
    "avg_cc_utilization": 0.1,
    "max_cc_utilization": 0.2,
    "avg_days_past_due_cc": 0,
    "num_credit_cards": 1,
    "pos_late_payments": 0,
    "avg_pos_days_past_due": 0,
}


async def _one_request(client: httpx.AsyncClient, url: str, headers: dict) -> tuple[float, int]:
    start = time.perf_counter()
    resp = await client.post(url, json=VALID_APPLICATION, headers=headers)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, resp.status_code


def summarize_results(results: list[tuple[float, int]], total_wall_seconds: float) -> dict:
    """Pure aggregation, kept separate from the network calls so it's testable
    without needing a live server (see tests/test_load_test.py)."""
    n_requests = len(results)
    latencies = sorted(r[0] for r in results)
    statuses = [r[1] for r in results]
    error_count = sum(1 for s in statuses if s != 200)

    return {
        "n_requests": n_requests,
        "total_wall_seconds": total_wall_seconds,
        "throughput_rps": n_requests / total_wall_seconds,
        "error_count": error_count,
        "error_rate": error_count / n_requests,
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": latencies[int(len(latencies) * 0.95)],
        "latency_ms_p99": latencies[int(len(latencies) * 0.99)],
        "latency_ms_max": latencies[-1],
    }


async def run_load_test(base_url: str, n_requests: int, concurrency: int) -> dict:
    headers = {"X-API-Key": os.environ.get("API_KEY", "")}
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(client):
        async with semaphore:
            return await _one_request(client, f"{base_url}/predict", headers)

    async with httpx.AsyncClient(timeout=30.0) as client:
        start = time.perf_counter()
        results = await asyncio.gather(*[bounded_request(client) for _ in range(n_requests)])
        total_wall_seconds = time.perf_counter() - start

    return {"concurrency": concurrency, **summarize_results(results, total_wall_seconds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()

    print(f"Load testing {args.url}/predict — {args.requests} requests, concurrency {args.concurrency}")
    result = asyncio.run(run_load_test(args.url, args.requests, args.concurrency))

    print(f"\nThroughput:  {result['throughput_rps']:.1f} req/s")
    print(f"Errors:      {result['error_count']}/{result['n_requests']} ({result['error_rate']:.1%})")
    print(f"Latency p50: {result['latency_ms_p50']:.1f} ms")
    print(f"Latency p95: {result['latency_ms_p95']:.1f} ms")
    print(f"Latency p99: {result['latency_ms_p99']:.1f} ms")
    print(f"Latency max: {result['latency_ms_max']:.1f} ms")


if __name__ == "__main__":
    main()
