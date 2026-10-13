# Superseded preliminary launch, 13 October 2026

This v1 is archived for provenance. Use the active [v2 study](../../../docs/followup_20261013.md).

CPU preflight (25688086) and GPU smoke (25688091) passed. The source/analysis review then found that a two-sided sign test plus a geometric-ratio magnitude criterion could support an improvement claim even when most seeds lose, if one extreme gain dominates the ratio. A new synthetic counterexample demonstrates this and v2 explicitly requires more curriculum wins than losses.

The 15 initially running development jobs in array 25688092 were cancelled after roughly one minute; no development model completed and no selection or confirmation ran. Remaining development slots and jobs 25688093, 25688094 and 25688095 were cancelled. See `superseded-slurm-status.txt` for the exact records. No partial result informed the inference correction or configuration selection. v2 uses different development seeds, confirmation seeds and evaluation banks and receives its own new source freeze.

The original 17 frozen source/protocol files, source commit, freeze, CPU report and runtime remain unchanged. The completed GPU report is preserved here. Partial weights/logs remain on the cluster in the superseded directory. No result from this launch is pooled into either the original completed study or v2.
