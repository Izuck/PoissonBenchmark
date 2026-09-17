# Poisson Benchmark

Two simple analytical tests, comparing **DEVSIM** and **scikit-fem**:

| Case | What it checks |
|---|---|
| **Regular capacitor** | Voltage through silicon and oxide; straight-line voltage in each layer. |
| **Sine** | A smooth 2D field, `sin(pi*x) * sin(pi*y)`; error should fall as the mesh gets finer. |

Both tools passed at **128, 256 and 512 subdivisions**, with five timed repetitions after one excluded warm-up.

[Measured results](benchmarks/runs/higher_20260917T040023_053900Z/report/benchmark_summary.md) · [Code walkthrough](benchmarks/CODE_MAP.md) · [Equations and timing](benchmarks/METHOD.md)

![Runtime and accuracy](benchmarks/runs/higher_20260917T040023_053900Z/report/scaling.png)

## Run it

Tested on **64-bit Windows 11 with Python 3.12.14**. From this repository folder, create a new local environment once:

```powershell
py -3.12 -m venv benchmarks/.venv
& benchmarks/.venv/Scripts/python.exe -m pip install -r benchmarks/requirements-lock.txt
```

Use Python 3.12.14 to match the recorded interpreter exactly. The lock file records the exact package versions used; the runner currently requires Windows.

Then double-click **Run Benchmark.cmd**, or run:

```powershell
& benchmarks/.venv/Scripts/python.exe benchmarks/baseline.py --quick
& benchmarks/.venv/Scripts/python.exe benchmarks/baseline.py
```

The first command checks the same two cases at 16, 32 and 64 subdivisions with one measured repetition. The second runs the full benchmark. Each creates a new folder in `benchmarks/runs/`; open its `report/benchmark_summary.html`.

## What is included

- Seven Python files: the runner, package adapters, analytical checks, and report helpers.
- Exact package versions, numerical settings, and a beginner-friendly code map.
- The completed run's raw timings, solver logs, hardware metadata, source snapshot, and plots.

The published run used an **Intel Core Ultra 9 285H, 31.6 GiB usable RAM, Windows 11**, one CPU computation thread, and no GPU. Timings include mesh setup, solving, interpolation, and saving the field; accuracy checks are outside the timer.

Generated voltage arrays (~144 MiB per full run) and environments are kept out of Git. Rerunning creates the arrays. Published logs replace the original computer's project-directory prefix with `<PROJECT_ROOT>`; numerical measurements and source snapshots are unchanged. The stored audit describes the original local run, including arrays that are not uploaded.

For a future solver or surrogate, use the same equations, boundaries, reference formulas, and error measurements. Compare runtime at comparable accuracy and state the timing scope. These are baseline checks, not a complete MOSFET simulation or a validation dataset covering general device behavior.
