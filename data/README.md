# Local Data Layout

The competition data is kept local and is not committed to git.

Expected structure:

```text
data/
  train_labels.csv
  sample_submission.csv
  train_images/
    000000.png
    ...
  test_images/
    000.png
    ...
```

Obtain data through the course's authorized distribution. No public download or
redistribution permission is provided here. Never commit images, labels, or
executed notebook outputs.

The active pipeline reads this repository-relative folder directly; `config.yaml`
does not relocate it. Training labels require `ID` (string, preserving leading
zeros) and `distance` (numeric). Training images follow label-row order and are
expected to be 300×300 RGB PNGs. Test PNGs are sorted lexicographically; the
writer assumes contiguous zero-padded IDs `000`, `001`, … . Compare output IDs
and order against the official sample submission, which the writer does not read.

Dataset counts, hashes, and version are not recorded here. Full benchmark
reproduction has not been verified without authorized data access.
