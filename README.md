# Curriculum learning for in-context linear regression

Source and results for the 8 October 2026 experiment: a curriculum over task dimension and supervised context length, compared with **the exact same realized training batches shuffled in time**, and with full difficulty from the start.

The 21 prospectively frozen source files are preserved byte for byte. This repository adds documentation, an archive verifier, CI, and explicitly exploratory review analyses around that source. It also retains the earlier rerun source snapshots for context.

## Reading guide

- [Results: tables, figures, caveats and raw evidence](docs/results.md)
- [Reproduction: recompute results or repeat GPU training](docs/reproduction.md)
- [Code walkthrough: exact execution path and implementing functions](docs/code_walkthrough.md)
- [Methods and actual configurations](docs/methods.md)

## New follow-up study

The [13 October prospective study](docs/followup_20261013.md) tests another dimension, separates dimension and context curricula, and gives every strategy an equal full-horizon optimizer search. Its source, protocol and submission evidence are separate from the completed results below. New outcomes are not available yet.

## What we ran

- 36 development models: three seeds × four optimizer configurations × three strategies, 50,000 updates.
- 60 confirmation models: 20 fresh paired seeds × three strategies, 125,000 updates.
- Clean Gaussian linear regression, 50 dimensions, 101 prediction positions; a 12-layer GPT-2-style Transformer, hidden size 256, eight heads; effective batch 256.
- The curriculum grows dimension from 5 to 50 and supervised context length during the first 25,000 updates. The shuffled control permutes the identical realized batches, including targets, loss masks and scales. Both then share the same 100,000-update full-difficulty tail.
- AdamW; the separate development search selected learning rate 0.0001 and raw loss for all confirmation strategies. The shared curriculum/shuffled configuration was selected using shuffled development performance.
- Python 3.12.3, PyTorch 2.9.0+cu126, Transformers 4.57.1, NumPy 2.3.4; NVIDIA A100 80GB PCIe; BF16 forward computation. Recorded package versions are included.

The hypothesis is supported **within this setup**, with a trade-off: curriculum improves clean predictions but is worse under noisy demonstrations. At the final checkpoint, it wins on clean error in 17/20 seeds and loses under noise SD1 in 19/20 and SD2 in 16/20. Final clean error has a geometric mean paired curriculum/shuffled ratio of 0.665. The prespecified clean learning-curve comparison wins in 16/20 seeds (exact sign-test p=0.01182). These are distinct endpoints.

The mean AUC-effect criterion depends on one slow shuffled run. Full-difficulty training split into ten learned and ten stalled models, so it is not a demonstrated strong baseline. New ratios, signed-rank tests and outlier sensitivity analyses are exploratory. See the [review brief](reports/confirmation/icl_curriculum_results_for_srikar_v2.pdf) and [interpretation notes](docs/interpretation.md).

## Start here: verify and recompute without training

Only Python's standard library is needed:

```bash
python3 tools/reproduce_results.py
```

`reproduce_results.py` runs the archive verifier and raw-metric reanalysis, then prints the clean/noisy results, frozen AUC comparison and baseline groups. `verify_archive.py` checks the raw archive checksum, archive integrity, source freeze, optimizer selection, all 1,140 development/confirmation metric records, pairing, and agreement with recorded run identities. The reanalysis recomputes scalar outcomes from raw metrics and checks them against the original summary.

For figures:

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.txt
.venv-analysis/bin/python analysis/make_review_figures.py data/confirmation_20261008.zip artifacts/figures
```

## Layout

| Path | Contents |
| --- | --- |
| `experiments/followup_20261013/` | New prospective study source, protocol, fixed task plan and cluster submission evidence. |
| `experiments/confirmation_20261008/` | Frozen confirmation source, original cluster scripts, recorded selection and preflight metadata; auxiliary source files have separate capture hashes. |
| `data/confirmation_20261008.zip` | Complete text/metric archive from the completed cluster run: 360 development and 780 confirmation records, manifests, training logs, source and metadata. |
| `analysis/` | Added post hoc review statistics and figures. |
| `reports/confirmation/` | Original summary and learning curves, revised review, all-seed tables and review audit. |
| `historical/initial_reruns/` | Earlier noise reruns, continuations, dimension studies, tests and cluster launch scripts. These are historical local snapshots. |
| `provenance/` | Source manifests, environment/runtime record, job IDs and verification reports. |
| `tools/` | Repository checks and safe staging for training reproduction. |
| `docs/` | Methods, reproduction, scope and interpretation. |

## Tests and training reproduction

Use a Linux CPU environment for scientific unit tests or a CUDA 12.6/BF16-capable GPU environment for the original training path. Detailed instructions and the distinction between archived results, numerical recomputation and fresh training are in [docs/reproduction.md](docs/reproduction.md).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
(cd experiments/confirmation_20261008 && ../../.venv/bin/python -m unittest -v test_confirmation test_robustness)
```

The original scripts use paths relative to their working directory. New work should be staged with `tools/stage_training.py`; it preserves the archived experiment and puts generated results outside tracked directories. The cluster scripts encode the original NTNU IDUN account, partition and hardware requests. Adapt those operational settings for another cluster in a separately labeled run; do not alter frozen source and present it as the historical experiment.

Training weights and binary schedules remain on the cluster; their recorded checksums are in the metric files. The archived original audit verified 720 confirmation checkpoints and 324 development checkpoints. This repository can reproduce numbers from raw text metrics and verify the code identities; downloading it does not repeat that binary checkpoint audit or guarantee bitwise equivalence on another device.

## Background and related work

This is a follow-up to *Robustness of In-context Learning via Curriculum Learning*, by Srikar Babu Gadipudi, Neil Pattanaik, Noah Lund Syrdal and Anders Vestrum ([original project](https://www.noahsyrdal.com/papers/ICLproject.pdf)). The new dimension/context experiment preserves the original negative noise-ordering result. The inspected original synthetic repository averages training loss across prediction positions; the paper's final-query-only wording should not be taken as verified historical run behavior.

- Garg et al. (2022): [What Can Transformers Learn In-Context?](https://arxiv.org/abs/2208.01066).
- Liu et al. (2025): [Probing In-Context Learning: Impact of Task Complexity and Model Architecture on Generalization and Efficiency](https://arxiv.org/abs/2505.06475).
- Shin & Alvarez-Melis (2026): [Curriculum Learning as Transport: Understanding Curricula with Wasserstein Geodesics](https://arxiv.org/abs/2609.09099). Its static control matches cumulative exposure by difficulty level; our control reshuffles the finite realized batch multiset.

Generic curriculum benefits for synthetic ICL are already known. The exact batch control and clean/noisy trade-off are candidate contributions for further review. Weaker implicit regularization is a hypothesis, not an established mechanism. The experiment implements regression Transformers independently using Hugging Face Transformers; source comments credit the earlier work and repositories.
