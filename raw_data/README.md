# Raw Data

The real-world demonstration CSV is stored as gzip-compressed CSV to keep the file below GitHub's regular file-size limit.

Use it directly with the plotting script:

```bash
python3 scripts/plot_quadhopper_ral_figures.py raw_data/quadhopper_deployed_data_20260704_160521.csv.gz --output-dir analysis_outputs/reproduced_outputs
```

`pandas.read_csv` reads `.csv.gz` files directly, so manual decompression is not required.
