# Data

Two public intrusion-detection datasets are used. **Neither the raw CICIoT2023 data nor our subsample is stored in this
repository** (see the licence notes below). Follow the steps here to rebuild exactly what we used.

## 1. NSL-KDD (5 classes: Normal, DoS, Probe, R2L, U2R)

| | |
|---|---|
| Files needed | `KDDTrain+.txt`, `KDDTest+.txt` |
| Put them in | `data/nsl_kdd/` |
| Original host | Canadian Institute for Cybersecurity (UNB), NSL-KDD page. The page now says the dataset is "no longer available" there; copies are widely mirrored (for example on Kaggle: search "NSL-KDD"). |
| Licence | The NSL-KDD page states that the dataset may be redistributed, republished and mirrored in any form, provided any use or redistribution cites the NSL-KDD dataset and the paper below. |

The two `.txt` files are small (about 20 MB together). They may be committed to `data/nsl_kdd/` **only together with the citation below**.

Citation:
> M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, "A Detailed Analysis of the KDD CUP 99 Data Set," in *Proc. IEEE Symp. Comput. Intell. Security Defense Appl. (CISDA)*, 2009.

Class mapping (attack labels to five groups) is in the code (`load_nsl` in `src/ci_exp.py`). Attack names outside the standard mapping are dropped.

## 2. CICIoT2023 (34 classes: 33 attacks + Benign)

| | |
|---|---|
| Source | Canadian Institute for Cybersecurity (UNB): <http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/> (project page: <https://www.unb.ca/cic/datasets/iotdataset-2023.html>) |
| What to download | The `CSV` directory only (one folder per class). **Do not download the PCAP folder.** |
| Licence | The dataset page states no licence, terms of use or redistribution rule. We therefore do **not** redistribute the data or our subsample; please download it yourself. |

Citation:
> E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, and A. A. Ghorbani, "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," *Sensors*, vol. 23, no. 13, art. 5941, 2023. <https://www.mdpi.com/1424-8220/23/13/5941>

### Rebuilding our subsample

The full dataset is large, so the experiments use a subsample: for each class folder we take the **first CSV file
(alphabetical) and keep its first 5,000 rows**. Smaller files are kept in full (Uploading_Attack has 1,252 rows,
Recon-PingSweep 2,262, XSS 3,846, Backdoor_Malware 3,218). The class name is the folder name. Run:

```bash
bash data/make_subsample.sh /path/to/CICIoT2023/CSV
```

This creates `data/ciciot_small/` (34 CSV files, about 160,000 rows in total, 39 numeric features) and `data/ciciot_small.zip`.

### Splits and preprocessing (as implemented in `src/ci_exp.py`)

- Per seed (0 to 9), each class is split at random: 30% test, then 10% of the remainder as validation, the rest training.
- Features are standardised with training-split statistics and clipped to [-10, 10]. Rows with NaN or infinity are dropped.
- Class-incremental stages (new classes only arrive): S1 Benign + 4 large DDoS floods; S2 8 other DDoS types;
  S3 DoS + Mirai; S4 Recon + MITM + DNS spoofing; S5 web and application attacks (rarest, hardest classes).

Note: this is a subsample from one capture file per class, so absolute scores are not comparable with results on the full dataset.
