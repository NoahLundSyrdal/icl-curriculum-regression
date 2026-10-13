# Active v2 submission, 13 October 2026

The corrected v2 was submitted from `/cluster/work/noahls/icl-rerun-20261005/followup_20261013_v2`. CPU preflight completed with exit 0:0: all 15 scientific tests and 150 production probes passed. GPU smoke completed successfully with exit 0:0 and all 50 production configuration checks passed. Fifteen development tasks are running, each allocated one exclusive A100 80GB GPU with oversubscription disabled. All later jobs retain success dependencies. No completed development or confirmation results are available yet.

| Stage | Slurm job | Verified initial status |
| --- | --- | --- |
| CPU preflight | 25688180 | Completed, 15 tests and all 150 production probes passed |
| GPU smoke | 25688191 | Completed, all 50 checks passed |
| Development, 150 tasks | 25688192 | Running, 15 concurrently; remaining tasks wait at array limit |
| Locked audited selection | 25688193 | Pending all development successes |
| Confirmation, 360 slots | 25688194 | Pending selection; 200..360 unique models |
| Final audit, summary and archive | 25688195 | Pending all confirmation successes |

Source commit: `ae467f9832abad47066f169ff628b5536bdc4005`. The freeze identifies 17 exact source/protocol files and was created at 2026-10-13T09:50:08.232641+00:00. Runtime is Python 3.12.3, Torch 2.9.0+cu126, Transformers 4.57.1; the full package freeze is included. The freshness audit checked 2487 prior manifest/metric records, including the superseded launch.

The v2 direction rule explicitly requires more curriculum wins than losses. A synthetic case with 19 losses and one extreme gain must fail even if its two-sided p-value and geometric ratio meet the other thresholds. The archived v1 GPU preflight passed, but its briefly running development jobs were cancelled and no selection/confirmation ran. All v2 seeds/banks are new, and no v1 output informs selection or inference. Original completed 8 October evidence is unchanged.

The arrays are capped at 15 concurrent GPUs. All same-dimensional strategies have the same development grid, full training duration, physical shapes and batch size. Boundary winners, full-difficulty learning/failure groups and all seeds will be reported. Tuning is fair within this finite grid and does not establish an optimal baseline. Initial throughput suggests roughly 1–2 weeks for the entire study at sustained GPU availability; actual queue availability and alias resolution determine elapsed time.

The successful final audit will generate `results/followup_20261013_v2.zip` and its checksum manifest with all text metrics, logs, code and provenance. Publishing the completed result archive in GitHub follows completion; weights and binary schedules remain on IDUN. This page is a dated submission snapshot, not live monitoring.

GitHub verification passed on the unchanged source: original 26 tests plus 15 new scientific tests, archive recomputation and source verification. See `repository-ci.json`. This commit adds completed GPU/launch records without changing any frozen source. The GPU preflight ran on NVIDIA A100-SXM4-80GB; requested production resources are A100 80GB and actual model hardware is recorded in every metric.
