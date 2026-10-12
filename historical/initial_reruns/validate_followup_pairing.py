#!/usr/bin/env python3
"""Check that reserved CPU thread count leaves initialization/episodes exact.

The original CLI used two Torch CPU threads; follow-up jobs reserve four.
This check verifies exact equality at production tensor sizes, including clean
targets and RNG transitions, rather than assuming thread-count independence.
"""
import dataclasses
from pathlib import Path
import sys

import torch
import run_experiment as base
import continue_experiment as tail


def main():
    records = []
    for seed in range(5):
        variants = []
        for threads in (2, 4):
            torch.set_num_threads(threads)
            cfg = base.preset_config("full", seed=seed, mode="clean", steps=20000, precision="bf16")
            base.seed_model(seed)
            model = base.RegressionTransformer(cfg)
            stream = base.EpisodeStream(seed)
            episodes = [stream.draw(cfg.microbatch, cfg.points, cfg.dims) for _ in range(4)]
            variants.append((model.state_dict(), episodes, stream.state_dict(), torch.get_rng_state()))
        a, b = variants
        if any(not torch.equal(a[0][key], b[0][key]) for key in a[0]):
            raise ValueError(f"Thread count changes initialization, seed {seed}")
        if any(not torch.equal(x, y) for left, right in zip(a[1], b[1]) for x, y in zip(left, right)):
            raise ValueError(f"Thread count changes a production episode, seed {seed}")
        if any(not torch.equal(a[2][key], b[2][key]) for key in a[2]) or not torch.equal(a[3], b[3]):
            raise ValueError("Thread count changes RNG transitions")
        records.append({"seed": seed, "initialization_bitwise_equal": True,
                        "four_microbatch_episodes_bitwise_equal": True,
                        "rng_transitions_bitwise_equal": True})
        print(f"seed {seed}: production initialization, episodes and RNG agree exactly", flush=True)
    base.atomic_json(Path("submissions/followup-thread-pairing.json"), {
        "thread_counts": [2, 4], "records": records, "torch": str(torch.__version__),
        "validator_sha256": tail.file_sha256(__file__), "base_sha256": tail.file_sha256(base.__file__)})


if __name__ == "__main__":
    main()
