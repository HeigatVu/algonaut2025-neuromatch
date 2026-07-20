# HMM Training Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Limit HMM fitting to 20 CPU workers and parallelize all candidate state-count restarts for faster model selection.

**Architecture:** Keep the notebook workflow and scientific settings intact. Replace the serial restart loop with one joblib process task per `(K, restart)`, cap each worker to one inner numerical thread, then group completed fits and retain the highest-likelihood restart for each K.

**Tech Stack:** Python 3.11, Jupyter notebook JSON, NumPy, hmmlearn 0.3.3, joblib 1.5.3, pytest 8.

## Global Constraints

- Modify only `src/notebook/get_data_one_episode.ipynb` and its focused test.
- Use candidate state counts 4 through 17 inclusive.
- Use at most 20 joblib workers and one BLAS/OpenMP thread per worker.
- Preserve five deterministic restarts, full covariance, 200 maximum iterations, tolerance `1e-2`, and best-training-likelihood restart selection.
- Do not change data loading, scaling, validation scoring, plotting, or output formats.
- Do not run the full 70-fit dataset workload during edit-time verification.
- Keep unrelated working-tree changes untouched.

---

### Task 1: Parallelize HMM restarts under a 20-worker ceiling

**Files:**
- Create: `tests/test_hmm_training_parallelism.py`
- Modify: `src/notebook/get_data_one_episode.ipynb`, cells `shared-hmm-config`, `shared-model-helpers`, `shared-selection`, and `shared-final-fit`
- Test: `tests/test_hmm_training_parallelism.py`

**Interfaces:**
- Consumes: `fit_hmm_restart(k, restart, values, lengths)` results shaped as `(k, restart, model_or_none, train_ll)`.
- Produces: `fit_best_hmms(k_values, values, lengths) -> dict[int, tuple[GaussianHMM, float]]`.

- [ ] **Step 1: Write the focused failing test**

Create `tests/test_hmm_training_parallelism.py`:

```python
import json
from pathlib import Path

import numpy as np


NOTEBOOK = Path("src/notebook/get_data_one_episode.ipynb")


def notebook_cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def cell_source(cell_id):
    return next(
        "".join(cell["source"])
        for cell in notebook_cells()
        if cell.get("id") == cell_id
    )


def test_parallel_training_configuration_and_usage():
    cells = notebook_cells()
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), NOTEBOOK.name, "exec")

    config = cell_source("shared-hmm-config")
    selection = cell_source("shared-selection")
    final_fit = cell_source("shared-final-fit")
    assert "K_VALUES = tuple(range(4, 18))" in config
    assert "N_JOBS = 20" in config
    assert "fit_best_hmms(K_VALUES" in selection
    assert "fit_best_hmms((selected_k,)" in final_fit


def test_parallel_helper_keeps_best_restart_and_caps_workers():
    calls = {}

    class FakeModel:
        def __init__(self, n_components, random_state, **kwargs):
            self.n_components = n_components
            self.random_state = random_state

        def fit(self, values, lengths):
            return self

        def score(self, values, lengths):
            return self.n_components - abs(self.random_state - 2001)

    class FakeParallel:
        def __init__(self, verbose):
            calls["verbose"] = verbose

        def __call__(self, tasks):
            return list(tasks)

    class FakeConfig:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    namespace = {
        "np": np,
        "warnings": __import__("warnings"),
        "GaussianHMM": FakeModel,
        "Parallel": FakeParallel,
        "delayed": lambda function: function,
        "parallel_config": lambda **kwargs: FakeConfig(**kwargs),
        "N_RESTARTS": 3,
        "N_JOBS": 20,
        "HMM_N_ITER": 200,
        "HMM_TOL": 1e-2,
        "RANDOM_SEED": 2000,
    }
    exec(cell_source("shared-model-helpers"), namespace)

    best = namespace["fit_best_hmms"](
        (4, 5), np.zeros((2, 1)), [2]
    )

    assert set(best) == {4, 5}
    assert all(model.random_state == 2001 for model, _ in best.values())
    assert calls == {
        "backend": "loky",
        "n_jobs": 6,
        "inner_max_num_threads": 1,
        "verbose": 10,
    }
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_hmm_training_parallelism.py -q
```

Expected: failures because the notebook still uses `range(4, 13)` and does not define `fit_best_hmms`.

- [ ] **Step 3: Add the joblib configuration**

In cell `shared-hmm-config`, keep `import joblib`, add:

```python
from joblib import Parallel, delayed, parallel_config
```

Replace the K configuration with:

```python
K_VALUES = tuple(range(4, 18))
N_RESTARTS = 5
N_JOBS = 20
```

- [ ] **Step 4: Replace the serial model helper**

In cell `shared-model-helpers`, replace `fit_best_hmm` with:

```python
def fit_hmm_restart(k, restart, values, lengths):
    """Fit one deterministic restart for one state count."""
    model = GaussianHMM(
        n_components=k,
        covariance_type='full',
        n_iter=HMM_N_ITER,
        tol=HMM_TOL,
        random_state=RANDOM_SEED + restart,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            model.fit(values, lengths)
        train_ll = model.score(values, lengths)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return k, restart, None, -np.inf

    if not np.isfinite(train_ll):
        return k, restart, None, -np.inf
    return k, restart, model, float(train_ll)


def fit_best_hmms(k_values, values, lengths):
    """Fit restarts in parallel and return the best model for each K."""
    k_values = tuple(k_values)
    n_jobs = min(N_JOBS, len(k_values) * N_RESTARTS)
    tasks = (
        delayed(fit_hmm_restart)(k, restart, values, lengths)
        for k in k_values
        for restart in range(N_RESTARTS)
    )
    with parallel_config(
        backend='loky', n_jobs=n_jobs, inner_max_num_threads=1
    ):
        fits = Parallel(verbose=10)(tasks)

    best = {}
    for k, _restart, model, train_ll in fits:
        if model is not None and (
            k not in best or train_ll > best[k][1]
        ):
            best[k] = (model, train_ll)

    failed = [k for k in k_values if k not in best]
    if failed:
        raise RuntimeError(f'All HMM restarts failed for K={failed}')
    return best
```

Leave `score_sequences` and `summarize_state_emotion` unchanged.

- [ ] **Step 5: Route candidate and final fitting through the parallel helper**

At the start of cell `shared-selection`, add:

```python
candidate_fits = fit_best_hmms(
    K_VALUES, train_values, train_lengths
)
```

Inside its K loop, replace the `fit_best_hmm` call with:

```python
candidate_model, train_ll = candidate_fits[k]
```

In cell `shared-final-fit`, replace the final `fit_best_hmm` call with:

```python
final_model, final_train_ll = fit_best_hmms(
    (selected_k,), final_train_values, final_train_lengths
)[selected_k]
```

- [ ] **Step 6: Run focused verification**

Run:

```bash
.venv/bin/python -m pytest tests/test_hmm_training_parallelism.py -q
```

Expected: `2 passed`.

Run:

```bash
.venv/bin/python -c 'import json; from pathlib import Path; p=Path("src/notebook/get_data_one_episode.ipynb"); cells=json.loads(p.read_text())["cells"]; [compile("".join(c["source"]), p.name, "exec") for c in cells if c["cell_type"]=="code"]; print("compiled", sum(c["cell_type"]=="code" for c in cells), "code cells")'
```

Expected: `compiled 21 code cells`.

- [ ] **Step 7: Review scope and commit the implementation**

Run:

```bash
git diff --check -- src/notebook/get_data_one_episode.ipynb tests/test_hmm_training_parallelism.py
git status --short
```

Confirm unrelated files remain unstaged. Then commit only the implementation files:

```bash
git add -f src/notebook/get_data_one_episode.ipynb tests/test_hmm_training_parallelism.py
git commit -m "perf: parallelize HMM training"
```

