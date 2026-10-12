#!/usr/bin/env python3
"""Measure full effective-batch throughput before launching a study."""
import gc
import json
import time

import torch
from run_experiment import EpisodeStream, RegressionTransformer, autocast_context, preset_config, seed_model


def main():
    assert torch.cuda.is_available(), "Run this on an allocated GPU."
    torch.set_num_threads(2)
    device = torch.device("cuda")
    precision = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    records = []
    for microbatch in (64, 256, 512):
        seed_model(0)
        cfg = preset_config("full", microbatch=microbatch, precision=precision)
        model = RegressionTransformer(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                     weight_decay=cfg.weight_decay)
        stream = EpisodeStream(0)
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for step in range(5):
            torch.cuda.synchronize()
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            accumulation = cfg.batch_size // cfg.microbatch
            for _ in range(accumulation):
                xs, clean, epsilon = stream.draw(cfg.microbatch, cfg.points, cfg.dims)
                xs, clean, labels = [x.to(device) for x in (xs, clean, clean + 0.5 * epsilon)]
                with autocast_context(device, precision):
                    loss = (model(xs, labels).float() - clean).square().mean()
                (loss / accumulation).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            durations.append(time.monotonic() - started)
        seconds = sum(durations[1:]) / 4
        record = {
            "gpu": torch.cuda.get_device_name(0), "precision": precision,
            "effective_batch": cfg.batch_size, "microbatch": microbatch,
            "seconds_per_optimizer_step": seconds,
            "estimated_10000_step_hours": seconds * 10000 / 3600,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        del model, optimizer, loss, xs, clean, labels
        gc.collect()
        torch.cuda.empty_cache()
    with open("benchmark_results.json", "w") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
