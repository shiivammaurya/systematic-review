# Continual learning for network intrusion detection: replay, regularization and retraining

Code and results for the paper *"Continual Learning for Network Intrusion Detection: A Structured Review of Replay,
Regularization, Distillation and Expansion"* (experiment section). Pure NumPy, no GPU.

## What is compared
- **Datasets:** NSL-KDD (5 classes, 4 stages, official train/test files) and CICIoT2023 (34 classes, 5 stages; first 5,000 rows of each class file).
- **Models:** neural network (one hidden layer, 64 ReLU units, Adam 1e-3) and logistic regression (SGD 0.05, L2 1e-4); batch 128, 5 epochs per stage.
- **Strategies (same for both models and datasets):** fine-tuning, EWC, replay (m = 200 per old class; m = 50 and 1000 as ablation), full retraining, class-balanced retraining.
- **Metrics:** macro-F1 over seen classes, forgetting F_T, per-class precision/recall/F1, samples processed, buffer memory; paired t-test and Wilcoxon over 10 seeds.
- **Budget-aware test:** choose between replay and retraining per stage (NSL-KDD budget 1.5 M samples, CICIoT2023 1.2 M); tau = 0.10 is the pre-specified threshold, other values are a sensitivity check.

## Run it
**Colab (easiest):** open `notebooks/CL_NIDS_Both_Datasets_Colab.ipynb`, run the cells top to bottom, upload `KDDTrain+.txt`, `KDDTest+.txt` and the CICIoT2023 subsample zip when asked. About 10-20 minutes with 10 seeds on the free tier (`SEEDS = 3` for a quick check).

**Locally:** `pip install -r requirements.txt`, then from `src/`:
```python
import runner
runner.run_dataset('nsl', '../data/NSL_KDD', seeds=10, procs=2, out_json='results_nslkdd.json')
runner.run_dataset('ciciot', '../data/CICIOT', seeds=10, procs=2, out_json='results_ciciot2023.json')
```
(the budget test is `budget.run_budget(...)`, see the notebook). Seeds change weight initialization, batch order, exemplar draws and
the Fisher subsample; on CICIoT2023 they also change the stratified 70/30 split.

## Results in this repository
`results/` holds the raw per-seed outputs (10 seeds) that the paper's tables are computed from, and the printed analyses
(`analysis_*.txt`). Runs are deterministic: the same seeds give the same numbers.

| | NSL-KDD NN | NSL-KDD LR | CICIoT2023 NN | CICIoT2023 LR |
|---|---|---|---|---|
| Fine-tune | 0.194 | 0.507 | 0.062 | 0.277 |
| Replay m=200 | 0.583 | 0.680 | 0.573 | 0.513 |
| Full retraining | 0.606 | 0.510 | 0.625 | 0.567 |

(final macro-F1, mean of 10 seeds)

## Notes and limits
- The CICIoT2023 files are the **first** rows of each class file, not a random draw.
- EWC strength is picked on the test data (favors EWC); two EWC forms are used: an explicit penalty for the network and a proximal step for logistic regression.
- Two datasets, one class order each, two small models.

## Citation
Please cite the paper (reference to be added) and the datasets: Tavallaee et al. (NSL-KDD, 2009); Neto et al., Sensors 23(13):5941 (CICIoT2023, 2023).
