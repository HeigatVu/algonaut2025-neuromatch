# Robust HMM Quality and Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude the uniquely corrupted subject-segment in memory, reject unhealthy HMM fits, and select the smallest healthy K within one validation standard error.

**Architecture:** Keep the existing shared seven-network HMM workflow and source HDF5 files unchanged. Add one conservative robust-artifact guard, filter the explicit `sub-02/s05e12b` exclusion while building group data, then augment candidate selection with training occupancy, extreme-mean diagnostics, and the one-standard-error rule.

**Tech Stack:** Python 3.11, Jupyter notebook JSON, NumPy, pandas, scikit-learn, hmmlearn 0.3.3, joblib 1.5.3, pytest 8.

## Global Constraints

- Modify only named code cells in `src/notebook/get_data_one_episode.ipynb` and `tests/test_hmm_training_parallelism.py`.
- Preserve all existing uncommitted notebook execution outputs.
- Never modify the source HDF5 files.
- Preserve seven network means, subject-specific scaling, Seasons 1-4 training, Season 5 validation, held-out S01E01, five restarts, full covariance, and the 20-worker limit.
- Exclude only `sub-02/s05e12b` from candidate validation and final training.
- Flag non-excluded TRs only when all seven network robust z-scores exceed 20.
- Define healthy candidates as `min_state_trs >= 50` and `max_abs_mean <= 10`.
- Select the smallest healthy K within one standard error of the best healthy validation mean.
- Write new outputs under `outputs/hmm_shared_s01e01_clean`.
- Do not add PCA, automatic interpolation, clipping, winsorization, or new dependencies.
- Do not run the full multi-season fit during edit-time verification.
- Keep unrelated working-tree changes unstaged.

---

### Task 1: Add robust data quality and K selection

**Files:**
- Modify: `src/notebook/get_data_one_episode.ipynb`, cells `shared-hmm-config`, `shared-data-helpers`, `shared-splits`, `shared-model-helpers`, `shared-selection`, `shared-selection-plot`, and `shared-final-fit`
- Modify: `tests/test_hmm_training_parallelism.py`
- Test: `tests/test_hmm_training_parallelism.py`

**Interfaces:**
- Consumes: seven-network arrays shaped `(n_tr, 7)`, subject IDs, `SegmentKey` objects, fitted `GaussianHMM` objects, and the existing selection table.
- Produces: `global_artifact_trs(values) -> np.ndarray`, `model_health(model, values, lengths) -> tuple[float, float, bool]`, and `select_k_one_se(selection_table) -> tuple[int, float]`.

- [ ] **Step 1: Write failing focused tests**

Extend `tests/test_hmm_training_parallelism.py` with:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler


def test_clean_configuration_is_explicit():
    config = cell_source("shared-hmm-config")
    assert "outputs/hmm_shared_s01e01_clean" in config
    assert "ARTIFACT_Z_THRESHOLD = 20.0" in config
    assert "MIN_STATE_TRS = 50.0" in config
    assert "MAX_ABS_STATE_MEAN = 10.0" in config
    assert "SegmentKey(5, 12, 'b')" in config


def test_global_artifact_guard_is_conservative():
    namespace = {
        "np": np,
        "NETWORKS": tuple(f"network-{index}" for index in range(7)),
        "ARTIFACT_Z_THRESHOLD": 20.0,
    }
    exec(cell_source("shared-data-helpers"), namespace)

    values = np.random.default_rng(0).normal(size=(100, 7))
    assert namespace["global_artifact_trs"](values).size == 0

    values[50] = 100
    assert namespace["global_artifact_trs"](values).tolist() == [50]

    constant = np.zeros((20, 7))
    assert namespace["global_artifact_trs"](constant).size == 0


def test_group_data_excludes_only_configured_subject_segment():
    calls = {}
    namespace = {
        "np": np,
        "SUBJECTS": ("sub-01", "sub-02"),
        "H5_PATHS": {"sub-01": "one", "sub-02": "two"},
        "DISCOVERED_BY_SUBJECT": {"sub-01": {}, "sub-02": {}},
        "EXCLUDED_SEGMENTS_BY_SUBJECT": {"sub-02": frozenset({"bad"})},
        "StandardScaler": StandardScaler,
    }
    exec(cell_source("shared-data-helpers"), namespace)

    def fake_load(path, discovered, segments):
        calls[path] = tuple(segments)
        arrays = [np.arange(14, dtype=float).reshape(2, 7) + index
                  for index, _ in enumerate(segments)]
        return arrays, list(segments)

    namespace["load_network_sequences"] = fake_load
    values, lengths, keys, _ = namespace["build_group_data"](("good", "bad"))

    assert calls == {"one": ("good", "bad"), "two": ("good",)}
    assert lengths == [2, 2, 2]
    assert keys == [
        ("sub-01", "good"), ("sub-01", "bad"), ("sub-02", "good")
    ]
    assert values.shape == (6, 7)


def test_model_health_and_one_se_selection():
    namespace = {
        "np": np,
        "MIN_STATE_TRS": 50.0,
        "MAX_ABS_STATE_MEAN": 10.0,
    }
    exec(cell_source("shared-model-helpers"), namespace)

    class HealthyModel:
        means_ = np.array([[0.0], [2.0]])

        def predict_proba(self, values, lengths):
            return np.tile([0.5, 0.5], (len(values), 1))

    assert namespace["model_health"](
        HealthyModel(), np.zeros((200, 1)), [200]
    ) == (100.0, 2.0, True)

    table = pd.DataFrame({
        "k": [4, 5, 6],
        "validation_mean": [0.89, 0.92, 1.00],
        "validation_se": [0.05, 0.05, 0.12],
        "healthy": [True, True, True],
    })
    assert namespace["select_k_one_se"](table) == (4, 0.88)

    table.loc[table["k"].eq(4), "healthy"] = False
    assert namespace["select_k_one_se"](table) == (5, 0.88)
```

Also extend `test_parallel_training_configuration_and_usage`:

```python
selection = cell_source("shared-selection")
final_fit = cell_source("shared-final-fit")
assert "min_state_trs" in selection
assert "max_abs_mean" in selection
assert "healthy" in selection
assert "select_k_one_se(selection_table)" in selection
assert "if not final_healthy:" in final_fit
```

- [ ] **Step 2: Run tests and confirm the new behavior is missing**

Run:

```bash
.venv/bin/python -m pytest tests/test_hmm_training_parallelism.py -q
```

Expected: the existing three tests pass and the new tests fail because the clean configuration and helpers do not exist.

- [ ] **Step 3: Add explicit clean-run configuration**

In `shared-hmm-config`, change the output root and add:

```python
RESULT_ROOT = PROJECT_ROOT / 'outputs/hmm_shared_s01e01_clean'

ARTIFACT_Z_THRESHOLD = 20.0
MIN_STATE_TRS = 50.0
MAX_ABS_STATE_MEAN = 10.0
EXCLUDED_SEGMENTS_BY_SUBJECT = {
    'sub-02': frozenset({SegmentKey(5, 12, 'b')}),
}
```

Keep `K_VALUES = tuple(range(4, 18))`, `N_RESTARTS = 5`,
`N_JOBS = 20`, and the existing HMM settings unchanged.

- [ ] **Step 4: Add artifact detection and subject-specific exclusion**

At the start of `shared-data-helpers`, add:

```python
def global_artifact_trs(values):
    """Return TRs with extreme simultaneous shifts in all networks."""
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = 1.4826 * mad
    robust_z = np.zeros_like(values, dtype=float)
    np.divide(
        np.abs(values - median),
        scale,
        out=robust_z,
        where=scale > 0,
    )
    return np.flatnonzero(
        (robust_z > ARTIFACT_Z_THRESHOLD).all(axis=1)
    )
```

In `load_network_sequences`, immediately after checking finite values, add:

```python
artifact_trs = global_artifact_trs(values)
if len(artifact_trs):
    raise ValueError(
        f'{h5_path.name}/{segment.task}: global artifacts at '
        f'TRs {artifact_trs.tolist()}'
    )
```

In `build_group_data`, filter before loading:

```python
excluded = EXCLUDED_SEGMENTS_BY_SUBJECT.get(subject, frozenset())
subject_segments = tuple(
    segment for segment in segments if segment not in excluded
)
sequences, segment_ids = load_network_sequences(
    H5_PATHS[subject],
    DISCOVERED_BY_SUBJECT[subject],
    subject_segments,
)
```

- [ ] **Step 5: Expose exclusions in notebook output**

In `shared-splits`, add:

```python
print('Excluded subject-segments:', {
    subject: [segment.task for segment in segments]
    for subject, segments in EXCLUDED_SEGMENTS_BY_SUBJECT.items()
})
```

- [ ] **Step 6: Add model-health and one-standard-error helpers**

Append to `shared-model-helpers` before `score_sequences`:

```python
def model_health(model, values, lengths):
    """Return occupancy, emission-extreme, and health diagnostics."""
    expected_trs = model.predict_proba(values, lengths).sum(axis=0)
    min_state_trs = float(expected_trs.min())
    max_abs_mean = float(np.abs(model.means_).max())
    healthy = (
        min_state_trs >= MIN_STATE_TRS
        and max_abs_mean <= MAX_ABS_STATE_MEAN
    )
    return min_state_trs, max_abs_mean, healthy


def select_k_one_se(selection_table):
    """Select the smallest healthy K within one validation standard error."""
    healthy = selection_table.loc[selection_table['healthy']]
    if healthy.empty:
        raise RuntimeError('No healthy HMM candidates')

    best = healthy.sort_values(
        ['validation_mean', 'k'], ascending=[False, True]
    ).iloc[0]
    cutoff = float(best['validation_mean'] - best['validation_se'])
    eligible = healthy.loc[healthy['validation_mean'].ge(cutoff)]
    return int(eligible['k'].min()), cutoff
```

- [ ] **Step 7: Record health and use the one-standard-error rule**

In `shared-selection`, after validation scoring, calculate:

```python
min_state_trs, max_abs_mean, healthy = model_health(
    candidate_model, train_values, train_lengths
)
```

Add these fields to every selection row:

```python
'min_state_trs': min_state_trs,
'max_abs_mean': max_abs_mean,
'healthy': healthy,
```

Replace direct maximum-validation selection with:

```python
selection_table = pd.DataFrame(selection_rows).sort_values('k')
selected_k, validation_cutoff = select_k_one_se(selection_table)
```

Extend each progress line with `healthy={healthy}`.

- [ ] **Step 8: Update the plot and validate the final fit**

In `shared-selection-plot`, add the cutoff to the validation panel:

```python
axes[1].axhline(
    validation_cutoff, color='tab:gray', linestyle=':',
    label='One-SE cutoff',
)
```

In `shared-final-fit`, after fitting the final model, add:

```python
final_min_state_trs, final_max_abs_mean, final_healthy = model_health(
    final_model, final_train_values, final_train_lengths
)
if not final_healthy:
    raise RuntimeError(
        'Final HMM is unhealthy: '
        f'min_state_trs={final_min_state_trs:.1f}, '
        f'max_abs_mean={final_max_abs_mean:.2f}'
    )
```

Add this metadata to the joblib payload:

```python
'excluded_segments_by_subject': {
    subject: tuple(segment.task for segment in segments)
    for subject, segments in EXCLUDED_SEGMENTS_BY_SUBJECT.items()
},
'artifact_z_threshold': ARTIFACT_Z_THRESHOLD,
'min_state_trs_threshold': MIN_STATE_TRS,
'max_abs_state_mean_threshold': MAX_ABS_STATE_MEAN,
'final_min_state_trs': final_min_state_trs,
'final_max_abs_mean': final_max_abs_mean,
```

- [ ] **Step 9: Run focused verification**

Run:

```bash
.venv/bin/python -m pytest tests/test_hmm_training_parallelism.py -q
```

Expected: all tests pass.

Run:

```bash
.venv/bin/python -c 'import json; from pathlib import Path; p=Path("src/notebook/get_data_one_episode.ipynb"); cells=json.loads(p.read_text())["cells"]; [compile("".join(c["source"]), p.name, "exec") for c in cells if c["cell_type"]=="code"]; print("compiled", sum(c["cell_type"]=="code" for c in cells), "code cells")'
```

Expected: `compiled 21 code cells`.

Run a read-only audit over Seasons 1-5 using the notebook detector. Expected:
`sub-02/s05e12b` is the only excluded sequence and every retained sequence
passes.

- [ ] **Step 10: Review scope and commit only task changes**

Run:

```bash
git diff --check -- src/notebook/get_data_one_episode.ipynb tests/test_hmm_training_parallelism.py
git diff --stat -- src/notebook/get_data_one_episode.ipynb tests/test_hmm_training_parallelism.py
git status --short
```

Because the notebook contains pre-existing uncommitted execution outputs,
stage only the task's source-code hunks and the focused test. Verify the cached
diff contains no output changes:

```bash
git diff --cached --check
git diff --cached --name-only
git diff --cached -- src/notebook/get_data_one_episode.ipynb tests/test_hmm_training_parallelism.py
```

Commit:

```bash
git commit -m "fix: reject degenerate HMM candidates"
```

