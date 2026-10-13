# Operational amendment: thesis takes priority, 13 October 2026

User authorized more ICL parallelism subject to thesis priority. Live `high` QoS permits 25 GPUs per user. `share-ie-idi` has a configured CPU quota of 5,766,000 hours, with roughly 4,993 reported CPU hours and 27,895 separately reported account GPU hours at the snapshot. These are shared department-account values, not Noah's personal allocation; the quota helper's CPU usage uses Slurm's reported `GrpTRESRaw` and excludes reservation usage. It is not a standalone guarantee of future scheduling.

Verified scheduling changes:

- Development array 25688192: task throttle 15 to 20, Nice 0 to 1,000,000.
- Confirmation array 25688194: task throttle 15 to 20, Nice 0 to 1,000,000.
- Selection 25688193 and summary/archive 25688195: Nice 0 to 1,000,000.
- The 20 ICL slots leave five slots below this user's 25-GPU QoS limit. The development and confirmation arrays cannot overlap because of their original success dependencies.
- Twenty development jobs were observed running after the change. The lower queue priority applies to remaining ICL scheduling; it does not preempt already running jobs.

This is user-limit headroom, not reservation of five physical GPUs. Pending thesis jobs submitted with normal Nice 0 retain priority advantage over ICL when other scheduling factors are comparable, but fair-share, resource fit, node availability and shared filesystem effects can still delay them. Both projects share the same user/account fair-share. If future thesis work requires more than five GPUs, ICL's pending concurrency must be reduced or held before that allocation; active GPU jobs are non-preemptible under GPUQ's current partition setting.

The original frozen protocol, task plan, Slurm source and submission receipt remain unchanged and still show the original cap of 15. The explicit live operational override is recorded in `resource-policy-20261013.json`, with before/after Slurm descriptions, commands and source-freeze hash. The exact operator script is preserved as `resource-policy-operator.py.txt`. No training source, seed, evaluation bank, optimizer search, model count, analysis endpoint or statistical rule changed. No thesis job or source was changed or resubmitted.

Estimated duration at continuous 20-GPU availability is roughly 6–9 days for the entire study, with queue/priority delays potentially extending it. Total compute does not drop because of increased concurrency. The final results archive includes these dated submission metadata files.
