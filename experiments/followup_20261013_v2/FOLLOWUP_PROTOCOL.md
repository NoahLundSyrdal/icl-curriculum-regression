# Prospective curriculum follow-up v2, 13 October 2026

This protocol is frozen by SHA256 before any new development outcomes. It is a local prospective freeze, not external preregistration. The completed 8 October experiment stays unchanged. This v2 supersedes the preliminary v1 launch after code review caught a missing direction requirement for two-sided significance. The first GPU preflight passed; 15 development tasks ran briefly and were cancelled before any development model completed or selection/confirmation began. None of those partial outcomes informs configuration selection or this source change. The change is motivated by a synthetic counterexample. v2 uses new seeds and banks and preserves the superseded source, freeze and cancellation record in Git history and the repository. All new results, failures and optimizer-boundary selections must be reported.

## Questions

1. Does easy-to-hard ordering help clean regression at target dimension 20 when compared with precisely the same batches shuffled?
2. Does that effect replicate at dimension 50, and does it arise with dimension-only or context-only curricula?
3. Do independently selected curriculum settings beat a full-difficulty model given the same optimizer tuning budget?
4. Does the clean/noisy trade-off persist and does the SD2 error peak move with task dimension? These noise/mechanism analyses are descriptive.

## Strategies and budgets

Ten strategies: joint dimension/context curriculum and its shuffled control at d=20; joint, dimension-only and context-only curricula and their respective shuffled controls at d=50; full-difficulty baselines at d=20 and d=50. The full-difficulty baseline is shared as a comparator within a dimension. Never count the same baseline or aliased run as independent evidence.

Every strategy receives the same development budget: three seeds (1700..1702), five learning rates (0.00001, 0.00003, 0.0001, 0.0003, 0.001), 125,000 updates each. That is 150 development models, all at the actual confirmation horizon. No candidate is stopped for poor performance and no holdout/noisy outcome enters selection. All use raw MSE, AdamW weight decay 0.01, gradient clipping 1, effective batch 256, microbatch 128, BF16 forward and FP32 loss/parameters. The grid now extends below the previous lower boundary, but does not guarantee an optimal baseline.

Fresh confirmation seeds are 2100..2119. Evaluation banks are 843131/843132 (development primary/holdout) and 843133/843134 (confirmation primary/holdout). Every evaluation uses 4,096 new regression episodes. Same-dimensional models share initialization and evaluation banks. These seed identities are audited against prior run manifests and primary/holdout metrics.

The architecture and batch construction are the unchanged verified training sources from the original confirmation: a causal 12-layer GPT-2-style Transformer, embedding 256, 8 heads, dropout zero; alternating numeric x/y vectors, predictions at x positions, no pretrained weights. d=20 uses 41 prediction positions and d=50 uses 101. Positional capacity remains 101. Physical shapes and token counts stay fixed within every same-dimensional comparison; shorter context means restricting supervised prefixes, not shrinking computation. Equal update budgets therefore imply equal model training FLOPs within dimension to the precision of the common implementation; wall time is also reported. Across dimensions, computational cost differs and absolute errors are not treated as the same task.

Curriculum starts at scheduled difficulty 5, holds for 1,000 updates, then increases linearly to full dimension at update 25,000. Joint curricula increase active dimensions and the number of supervised positions (2*scheduled+1). Dimension-only changes coordinates while supervising all positions; context-only uses full coordinates while increasing supervised positions. Low-dimensional targets are multiplied by sqrt(full_dimension/active_dimension), keeping expected target variance constant. Mean loss includes k=0.

Shuffled controls permute precisely the realized first 25,000 batch IDs with seed+9000003. Difficulty, masks, labels and supervision are functions of batch ID, not current update. Thereafter all same-dimensional strategies have an identical 100,000-update full-difficulty tail. Exact multiset matching applies within each component's curriculum/shuffled pair, not between distinct components or versus full difficulty.

## Two distinct comparison panels

Shared-optimizer ordering: for each curriculum component/dimension, choose the LR minimizing the arithmetic mean of all six curriculum/shuffled development final clean NMSEs (three seeds each). This symmetric choice does not favor either ordering. Both confirmation arms use that same selected LR. It isolates the effect of order under identical batches, optimizer and compute.

Independently tuned policy: each of the ten strategies chooses its own LR minimizing its three development final clean NMSEs. Every strategy has the same candidate grid and full training budget. Compare tuned curriculum against tuned full difficulty. Tuned curriculum versus tuned shuffled is reported descriptively as a practical comparison and does not isolate ordering if selected LRs differ.

Ties break by the fixed LR grid order. If a selected configuration lies at a boundary, report that limitation; no automatic extension or confirmation-driven retuning. The 360 fixed confirmation slots include both panels and two baselines per seed. Identical seed/strategy/LR configurations are aliased to the earliest slot, so only 200..360 distinct models train. Aliases are not new observations and cannot inflate sample size. Resolution is frozen before confirmation starts.

## Evaluation and inference

Full clean Gaussian regression: x,w ~ N(0,I_d), clean y=x dot w, normalized clean-target squared error / d. Context labels receive Gaussian noise SD0, SD1 or SD2 in raw label units; scoring targets stay clean. The fixed raw SD levels replicate the original follow-up but represent different noise-to-signal ratios at d=20 and d=50. Do not interpret a cross-dimensional difference as solely a dimension effect.

Final late-context NMSE averages k=d..2d at update 125,000. Clean learning-curve AUC is the trapezoidal mean over 25k,30k,40k,50k,75k,100k,125k updates, and is secondary descriptive. This is a new prospectively chosen final endpoint based on earlier results, not a retrospective replacement for the original AUC endpoint.

Primary confirmatory comparison: joint d20 shared-optimizer curriculum versus shuffled, final clean NMSE, across all 20 paired seeds. A support claim requires two-sided exact paired sign-test p<0.05 (absolute difference ties <=1e-8), more curriculum wins than losses, geometric mean paired error ratio <=0.90, and arithmetic mean curriculum NMSE <=0.25. The sign test concerns win probability; the ratio threshold is an observed practical magnitude criterion, not an independently established population effect bound.

Only if the full primary criterion passes may seven secondary clean final comparisons support confirmatory claims: shared-optimizer ordering for joint d50, dimension-only d50 and context-only d50; independently tuned curriculum versus the tuned full-difficulty baseline for all four groups. Use Holm's correction across these seven two-sided sign tests, adjusted p<0.05, with the same direction, ratio and learned-model criteria. If the primary fails, all seven remain descriptive. This fixed sequence with Holm controls the family of confirmatory claims.

Report per-seed values, arithmetic means, medians, geometric paired ratios, all win/loss/tie counts, and descriptive paired bootstrap intervals (10,000 resamples, seed 921733). Leave-one-seed-out ratios are sensitivity diagnostics only; no seed may be excluded because of its result. Report full-difficulty learning/failure counts using final NMSE<=0.25, rather than hiding bimodality in a grand mean.

Noise evaluations, independent final holdout, peak ratios, AUC, tuned curriculum/shuffled comparisons, bootstrap intervals and component interactions are descriptive, without additional confirmatory significance claims. The holdout evaluates the same trained cohort, not an independent training replication. Peak diagnostic: average NMSE over k=d-2..d+2 divided by average over k=d-12..d-8; include entire curves and whether this describes a true local peak. No fitted ridge penalties or assertion of neural OLS/ridge algorithm identity.

## Execution, audit and recovery

Run CPU tests plus production batch probes and seed audit first. GPU smoke exercises every LR/treatment at both dimensions, paired initialization, bitwise restart, causal future/own-label mutation, and independently constructed prompts without the query target. All development jobs depend on successful smoke; selection depends on all development successes; all confirmation slots depend on locked audited selection; final summary depends on every confirmation slot succeeding.

Infrastructure interruptions may resume only the unchanged task with its optimizer, batch order and RNG. Do not rerun a poorly performing completed model. Nonfinite training is recorded as a failure and blocks selection/claims rather than being silently discarded. No scientific outcome-dependent budget expansion is permitted. Preserve source/plan hashes, runtime package freeze, submission IDs, all logs/manifests/metric JSONs, batch schedules, rolling checkpoint and baseline/boundary/final snapshots. CPU audits validate retained checkpoint hashes and all raw metrics. Intermediate evaluation hashes are preserved, but intermediate weight binaries beyond baseline/boundary/final are not retained.

Use the existing share-ie-idi account, high QoS, A100 80GB GPUQ resources and at most 15 simultaneous GPUs. Development and confirmation cannot overlap due to dependencies. Each GPU task requests four CPU cores, 24GB RAM and a 16-hour wall limit. The worst-case plan is 150 development plus 360 unique confirmation models; aliases may reduce confirmation to 200. Budget is prospectively capped by these tasks, not by desired outcomes. A failure blocks downstream jobs until transparently resolved. The final successful summary automatically creates results/followup_20261013_v2.zip with raw text metrics, logs, source, selection, freeze and submission evidence plus a checksum manifest. No credentials, virtual environment or model weights enter that archive.
