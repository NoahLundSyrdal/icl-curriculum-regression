# Submission status, 13 October 2026

The full study was submitted on IDUN from `/cluster/work/noahls/icl-rerun-20261005/followup_20261013`. CPU preflight completed with exit code 0:0. GPU smoke is running; downstream development/selection/confirmation/summary jobs wait on success dependencies. No new development or confirmation results are available yet.

| Stage | Slurm job | Initial verified status |
| --- | --- | --- |
| CPU scientific preflight | 25688086 | Completed, 14 tests and all 150 production probes passed |
| GPU smoke | 25688091 | Running |
| Development, 150 tasks | 25688092 | Pending GPU smoke success |
| Audited selection | 25688093 | Pending all development successes |
| Confirmation, 360 slots | 25688094 | Pending locked selection; 200..360 actual models |
| Final audit, summary and archive | 25688095 | Pending all confirmation successes |

The freeze has 17 SHA256-identified source/protocol files, created at 2026-10-13T09:37:17.706459+00:00. Exact source commit: `7eb7438290c5993eca99b0a783c6d1f792f42899`. The subsequent repository commit adds the unchanged live cluster receipts. Runtime: Python 3.12.3, Torch 2.9.0+cu126, Transformers 4.57.1; full environment freeze is included. All 2257 prior manifest/metric records passed the new seed and evaluation-bank freshness audit.

The arrays use at most 15 concurrent GPUs. Same-dimensional strategies receive equal grids, training updates and model shapes. Selected boundary LRs and baseline failures will be reported; tuning is fair within the frozen finite search, not a claim of optimality. Actual completion time depends on measured production throughput and queue availability; this is a multi-day study.

Source and submission snapshots are preserved here. Results are generated on IDUN; final text metrics, logs, source and provenance will be archived automatically after a successful final audit. Publishing that result archive in this repository is a later step, after the runs complete. This status is a dated snapshot, not live monitoring.
