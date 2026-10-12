ICL curriculum confirmation - IDUN launch, 8 October 2026

Authoritative frozen design: CONFIRMATION_PROTOCOL.txt + CONFIRMATION_TASKS.json.
This replaces the earlier two-arm paper-readiness proposal with a three-arm
study aligned with the broader curriculum hypothesis. Primary: curriculum vs
independently tuned hard-only. Gated secondary: ordering vs matched shuffled.
36 development +60 confirmation models, 20 fresh paired confirmation seeds.
Maximum 15 concurrent GPUs; no array overlap. No message to Srikar.

Remote root:
/cluster/work/noahls/icl-rerun-20261005/confirmation_20261008

Jobs:
CPU preflight 25629201 COMPLETED 0:0 (26 tests and 96 batch probes).
GPU scientific smoke 25629208 COMPLETED 0:0 (all optimizer/arm, restart and causal checks).
Development array 25629209 (36 tasks).
Strict selector 25629210.
Confirmation array 25629211 (60 tasks).
Strict final summary 25629212.
Each subsequent stage depends on afterok; a failed stage stops the chain.

Read-only status after authenticating:
squeue -u noahls -o "%.18i %.24j %.8T %.10M %R"
sacct -j 25629201,25629208,25629209,25629210,25629211,25629212 -X -n -P --format=JobID,State,ExitCode,Elapsed

When selector completes:
submissions/selection.json preserves every development clean AUC/ranking and
its source freeze. submissions/confirmation_resolved_tasks.json fixes the
selected optimizer for every confirmation model. Both hashes enter identities.

When summary completes:
results/confirmation/summary.json
results/confirmation/RESULTS.txt
results/confirmation/learning_curves.csv
Individual curves/holdouts and all full model/Adam/RNG checkpoints stay in the
seed/variant directories. This source archive omits weights and production
outcomes not yet generated; it is not a finished results archive.

Environment is recorded in submissions/environment.txt; tested runtime:
Python 3.12.3, PyTorch 2.9.0+cu126, Transformers 4.57.1.
The unchanged older trainer and its unit tests are included for reproduction.
Do not submit the jobs again: the source freeze/submission guards prevent blind
duplicates. Inspect saved job IDs before any recovery. Any technical retry
must preserve the same seed/config/source and remaining fixed update budget.
Stalled learning is an outcome, not a technical failure to remove.

The study can support or fail to support the theory. It cannot guarantee a
positive result, formal proof, general LLM claim, or paper acceptance.
