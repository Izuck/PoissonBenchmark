# Run the two cases

Follow the one-time environment setup in the [main README](../README.md), then double-click `Run Benchmark.cmd` in the repository root.

For a small check of the same regular and sine cases:

```powershell
& benchmarks/.venv/Scripts/python.exe benchmarks/baseline.py --quick
```

Each run saves a new folder under `benchmarks/runs/`. Open its `report/benchmark_summary.html`.

[Code explained](CODE_MAP.md) · [Recorded results](runs/higher_20260917T040023_053900Z/report/benchmark_summary.md)
