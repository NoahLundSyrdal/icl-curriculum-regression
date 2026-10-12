# Interpretation and scope

The strongest comparison for ordering is curriculum versus exact same-batches shuffled, at the common configuration selected using shuffled development results. The full-difficulty baseline had only a small development grid, a chosen learning rate at the grid's lower boundary, and bimodal outcomes. It should not be called a proven strong or optimal baseline. Machine keys such as `primary_curriculum_vs_tuned_hard` in frozen code and original summaries are retained for provenance; they are not the wording we use in the revised scientific interpretation.

Clean final error and clean AUC are different outcomes. The frozen AUC analysis retains all 20 seeds, uses a two-sided exact sign test, and required observed mean improvement at least 0.02 plus curriculum final mean NMSE at most 0.25. Its mean gain of 0.02570 falls to 0.00832 when seed 1017 is removed post hoc; the median paired gain is 0.00748. The mean threshold's sensitivity must be reported.

Added signed-rank tests and ratios are exploratory. A signed-rank test assumes more about the paired null distribution than the sign test. Training seeds, not evaluation prompts, are the replication units. Interquartile figure bands are spread across seeds, not confidence intervals.

The noise counts refer to final clean-target error with noisy demonstration labels. The independent final holdout repeats those counts, but uses the same trained models and is not another independent training cohort. Neither the noisy evaluations nor the post hoc peak-ratio comparison reverses the original negative increasing-noise result.

Under SD2 noise the larger error peak near the task dimension is consistent with weaker implicit regularization. At SD1, the ratio of average error over k=48..52 to k=38..42 is larger for curriculum but below one in both groups; that indicates less decline rather than proving a local peak. Fragile fitted-ridge estimates and unverified reference overlays are omitted. No claim is made that either neural model implements OLS or ridge.

This is one function class, architecture, dimension and finite optimizer grid. Publication novelty, historical run provenance and generalization need further review. The revised brief gives more detailed numbers and citations.
