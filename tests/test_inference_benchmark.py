from types import SimpleNamespace

import torch

from scripts.benchmark_gars_inference import _measure_mode


class _TinyLogitModel:
    def eval(self):
        return self

    def __call__(self, input_ids):
        values = input_ids.float().mean(dim=1, keepdim=True)
        return SimpleNamespace(logits=torch.cat((values, -values), dim=1))


def test_measure_mode_reports_synchronized_latency_and_throughput() -> None:
    model = _TinyLogitModel()
    batches = [
        {"input_ids": torch.ones((2, 4), dtype=torch.long)},
        {"input_ids": torch.full((2, 4), 2, dtype=torch.long)},
    ]

    result = _measure_mode(
        model=model,
        batches=batches,
        batch_size=2,
        device=torch.device("cpu"),
        warmup_steps=2,
        timed_steps=4,
    )

    assert result["timed_steps"] == 4
    assert result["timed_examples"] == 8
    assert result["mean_batch_latency_ms"] > 0
    assert result["median_batch_latency_ms"] > 0
    assert result["p90_batch_latency_ms"] > 0
    assert result["throughput_examples_per_second"] > 0
