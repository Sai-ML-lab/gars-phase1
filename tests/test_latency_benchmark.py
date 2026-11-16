from scripts.benchmark_inference_latency import _summarize


def _run(seed: int, method: str, unmerged_latency: float, merged_latency: float) -> dict:
    return {
        "seed": seed,
        "method": method,
        "unmerged": {
            "1": {
                "latency_ms_median": unmerged_latency,
                "examples_per_second": 1000.0 / unmerged_latency,
            }
        },
        "merged": {
            "1": {
                "latency_ms_median": merged_latency,
                "examples_per_second": 1000.0 / merged_latency,
            }
        },
    }


def test_latency_summary_computes_paired_speedup_against_parent() -> None:
    runs = [
        _run(42, "parent", 10.0, 8.0),
        _run(42, "gars", 8.0, 6.0),
        _run(123, "parent", 12.0, 9.0),
        _run(123, "gars", 9.0, 7.0),
    ]

    summary = _summarize(
        runs,
        seeds=[42, 123],
        methods=["parent", "gars"],
        batch_sizes=[1],
    )

    unmerged = summary["unmerged:gars:batch_1"]
    assert unmerged["speedup_vs_parent_by_seed"] == {"42": 1.25, "123": 12.0 / 9.0}
    assert unmerged["mean_speedup_vs_parent"] == (1.25 + 12.0 / 9.0) / 2.0

    merged = summary["merged:gars:batch_1"]
    assert merged["speedup_vs_parent_by_seed"] == {"42": 8.0 / 6.0, "123": 9.0 / 7.0}


def test_latency_summary_works_without_parent_baseline() -> None:
    summary = _summarize(
        [_run(42, "gars", 8.0, 6.0)],
        seeds=[42],
        methods=["gars"],
        batch_sizes=[1],
    )

    assert summary["unmerged:gars:batch_1"]["mean_speedup_vs_parent"] is None
