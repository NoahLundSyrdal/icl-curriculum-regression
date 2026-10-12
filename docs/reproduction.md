# Reproduction

## 1. Verify the archived observations and source

Clone the private repository using an account with access:

```bash
gh repo clone NoahLundSyrdal/icl-curriculum-regression
cd icl-curriculum-regression
python3 tools/reproduce_results.py
```

The last command prints the main numerical results after checking the archive and independently recomputing outcomes. Expected tables and precise endpoint definitions are in [results.md](results.md).

To run the two underlying checks separately, from the repository root:

```bash
python3 tools/verify_archive.py
python3 analysis/recompute_review.py data/confirmation_20261008.zip artifacts/review_recomputed.json
```

The first command checks the prospectively frozen 21 files, auxiliary captured files, optimizer resolution, archive SHA256, raw identities, 96 trained-model manifests, and 1,140 evaluations. It writes a verification report under `artifacts/`, which is ignored by Git. The second recomputes each scalar AUC/final/holdout outcome directly from raw metric files, verifies the summary and produces the post hoc ratios/counts/tests. Neither command imports Torch or trains a model.

The complete archive was captured from the completed run on 12 October 2026. Source freeze: 8 October 2026 at 09:32:21 UTC. It includes 36 development and 60 confirmation models. Original production job IDs and recorded runtime are in `provenance/REPOSITORY_PROVENANCE.json`. The revised review analyses were added after observing the data; they do not modify the frozen protocol.

## 2. Plot the recorded data

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.txt
.venv-analysis/bin/python analysis/make_review_figures.py data/confirmation_20261008.zip artifacts/figures
```

The figure generator uses every seed, with median curves, seed interquartile bands and paired ratios. The PDF and original published review figure are archived outputs; plotting dependencies are new review tooling, not the training runtime. The report's plotted curves have been reproduced during packaging.

## 3. Run scientific unit tests

For Linux x86-64 with Python 3.12.3:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
(cd experiments/confirmation_20261008 && ../../.venv/bin/python -m unittest -v test_confirmation test_robustness)
```

The CPU wheel differs from the actual CUDA training distribution. Tests cover paired batches, target variance, chronological controls, causal/query-target absence, restart state, selection and inference rules. These tests are useful checks, not an independent confirmation of the scientific effect. GitHub CI also verifies the source/data archive and recomputes the review statistics. Historical tests are preserved for their own experiment directories; they are not the confirmation test suite.

## 4. Stage fresh training using the original code

For a BF16-capable CUDA GPU, install `requirements-gpu.txt`. The complete original package list is `experiments/confirmation_20261008/submissions/environment.txt`; the original CPU module was `Python/3.12.3-GCCcore-13.3.0`. Linux/CUDA wheels and their availability on another machine are environmental constraints. Match the recorded GPU/software stack as closely as possible; a different machine is not automatically bitwise equivalent.

The staging utility refuses a nonempty destination and copies source files byte for byte:

```bash
python3 tools/stage_training.py --mode confirmation --destination runs/confirmation-reproduction
```

For a single original confirmation task, after preparing a GPU environment:

```bash
cd runs/confirmation-reproduction
python3 -m venv .venv
.venv/bin/python -m pip install -r ../../requirements-gpu.txt
.venv/bin/python confirmation_experiment.py --cohort confirmation --task-index 0
```

For all 60 confirmation models on IDUN, after creating that staged `.venv`, use the unchanged training array and then the summary dependency:

```bash
mkdir -p logs
icl_array_job=$(sbatch --parsable --account=share-ie-idi --qos=high confirmation_train.slurm)
sbatch --account=share-ie-idi --qos=high --dependency=afterok:${icl_array_job} --kill-on-invalid-dep=yes confirmation_summary.slurm
```

These commands submit GPU work; they are not needed to recompute archived numbers. The summary runs only if the full array succeeds and requires the checkpoints produced by the new training. Outside IDUN, use the single-task entry point in your own scheduler and run the original summary after all 60 tasks finish.

This mode copies the original freeze plus the locked development selection and resolved confirmation tasks. It reproduces the confirmation configurations conditioned on that recorded selection. Run task indices 0 through 59 to repeat the full confirmation cohort. The task table fixes seeds/configurations; a rerun with those seeds is reproduction, not a fresh independent statistical confirmation. Use a new protocol and seeds for a new independent study.

For the complete original development/selection pipeline on IDUN, create the environment **inside the staged run**: every Slurm script expects `.venv/bin/python` relative to that directory. Start from the repository root:


```bash
python3 tools/stage_training.py --mode full --destination runs/full-reproduction
cd runs/full-reproduction
module purge
module load Python/3.12.3-GCCcore-13.3.0
python3 -m venv .venv
.venv/bin/python -m pip install -r ../../requirements-gpu.txt
.venv/bin/python submit_confirmation.py --preflight-only
# After CPU preflight completes successfully:
.venv/bin/python submit_confirmation.py
```

`full` copies only source and no existing submissions/results. The original submitter creates a new freeze and launches CPU checks, GPU smoke checks, 36 development jobs, selection, 60 confirmation jobs, and summary with success dependencies. This can use hundreds of GPU-hours. These commands are reproduction instructions; packaging the repository does not execute them. The frozen submitter has the original allocation and QoS; check their applicability before intentionally launching work. Changed source creates a different run identity and must not be labeled the original experiment.

The original selector and summary audit check checkpoint files, binary schedules and train logs. Those binaries remain on the cluster and are not in GitHub. To audit a fresh training run, keep its retained checkpoints and run the original summary against the new outputs. Merely extracting metric JSON and rerunning the original strict summary cannot substitute for this checkpoint audit.

## 5. Historical source

`historical/initial_reruns/` contains the earlier local source snapshots for noise reruns, continuations, dimension pilots and long runs, with their protocols and tests. Packaging hashes are in `provenance/HISTORICAL_SOURCE_MANIFEST.json`; these do not imply those files were prospectively frozen with the later confirmation. Their raw result history and model weights are not included in the current confirmation archive. The current paper-review claims come from the later 20-seed confirmation and are not pooled across exploratory studies.
