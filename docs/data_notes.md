# Data Notes

Dataset used here: Avast-CTU Public CAPEv2 Dataset.

For routine experiments, the reduced reports archive is enough. It keeps the
same family labels, malware types, and dates from `public_labels.csv`, while
being much easier to manage locally than the full archive.

I would only switch to the larger archive if a later analysis needs report
fields that are not present in the reduced version.

Accepted inputs for `scripts/01_ingest.py`:

- `public_labels.csv`
- `public_small_reports.zip`
- `avast_ctu_reduced.zip`
- `1.zip`
- `2.zip`

Sources:

- https://github.com/avast/avast-ctu-cape-dataset
- https://arxiv.org/abs/2209.03188
