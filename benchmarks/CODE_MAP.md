# How the two-case test works

The benchmark asks **two existing solvers** the same questions, checks their answers, and records the time. We are not writing a new numerical solver here.

```text
baseline.py                  runs the experiment
    ├─ adapters.py           sine case + calls to DEVSIM/scikit-fem
    ├─ moscap.py              regular capacitor + calls to the same tools
    ├─ problems.py           capacitor settings and known answer
    ├─ checks.py             compares results with the known answers
    └─ scaling.py            makes the table, graph and short report
```

**Start with `baseline.py`.** Its settings are at the top:

```python
CASES = ('moscap', 'sine')     # regular capacitor, then sine
TOOLS = ('devsim', 'scikit-fem')
MESHES = (128, 256, 512)      # subdivisions per side
REPEATS = 5
```

There are two main functions:

- **`run()`** is the organizer. It starts one worker at a time and saves the hardware/software settings.
- **`worker()`** does the work: build the mesh → solve → save the voltage → check the answer. It warms up once, then repeats five times.

Each solver wrapper exposes the same operations: `solve()`, `sample()`, `stats()`, and `close()`. That keeps the experiment the same even though each package has different commands.

**The two cases**

| Case | Known answer | What it checks |
|---|---|---|
| Regular capacitor (`moscap`) | Straight voltage profile within each material layer | Correct boundaries/material interface; runtime as the mesh grows |
| Sine (`sine`) | `sin(pi*x) * sin(pi*y)` | A varying 2D field; error should fall about fourfold when subdivisions double |

**What gets saved**

Each case/tool/mesh has a small settings dictionary, a log and voltage arrays. Arrays are indexed **[y, x]**. `raw_results.csv` contains the measurements; `report/benchmark_summary.html` explains them.

The timer covers mesh setup, solving and saving the same 512 × 512 output. Accuracy checking happens afterwards. A separate check on the actual mesh confirms that the output grid does not hide fine-mesh errors.

**Run both cases**

Double-click `Run Benchmark.cmd`, or:

```powershell
& benchmarks/.venv/Scripts/python.exe benchmarks/baseline.py
```

For a small check of these same two cases, add `--quick`.
