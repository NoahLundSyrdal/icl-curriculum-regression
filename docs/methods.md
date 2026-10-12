# Methods at a glance

The full method is the unchanged `experiments/confirmation_20261008/CONFIRMATION_PROTOCOL.txt` plus its task plan and recorded selection. The table below describes the actual confirmation configurations, not the older defaults in `run_experiment.py`.

| Setting | Actual confirmation |
| --- | --- |
| Training task | Clean Gaussian linear regression; x,w ~ N(0,I), y=x dot w |
| Target dimension | 50 |
| Sequence | 101 prediction positions, alternating x/y tokens |
| Transformer | GPT-2-style causal model, 12 layers, 256 embedding, 8 heads, zero dropout |
| Effective batch / microbatch | 256 / 128 |
| Optimizer | AdamW, LR 0.0001, weight decay 0.01, gradient clipping 1.0 |
| Precision | BF16 forward, FP32 parameters/loss |
| Curriculum | Dimension 5 to 50 over first 25k updates; supervises 2d+1 prediction positions |
| Target scaling | sqrt(50/d) during low-dimensional batches; expected target variance stays 50 |
| Shuffled control | Exact realized first-25k batch multiset in permuted order |
| Common tail | Identical full-difficulty batch sequence for remaining 100k updates |
| Physical shape | 50 input coordinates and 101 positions throughout each arm |
| Training objective | Mean squared clean-target loss over supervised prediction positions, including k=0 |
| Confirmation seeds | 1000..1019, paired initialization and base examples |
| Evaluation banks | 731841 primary, 731842 final holdout; 4,096 episodes each |
| Noise | Gaussian demonstration-label noise with raw SD 0, 1, 2; clean query target |
| Clean AUC | Trapezoidal mean NMSE over 25k,30k,40k,50k,75k,100k,125k updates |
| Late-context error | Mean clean-target squared error / 50 over k=50..100 |

The development cohort uses seeds 600..602, banks 731839/731840, four optimizer configurations, and 50k updates. Shuffled development performance chooses the shared curriculum/shuffled optimizer, while full difficulty selects its own setting from the same grid. Both selected the low-LR raw-loss setting; it ranked third for curriculum development performance.

The adapter `confirmation_experiment.py` applies the actual task overrides and records identities. `run_experiment.py --preset full` alone is an earlier 20-dimensional noise experiment and does not run the confirmed study. Follow the confirmation entry point and frozen task list.
