# Shared Concatenated HMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the notebook's subject-by-network HMM section with one readable shared HMM trained on concatenated subject sequences while keeping S01E01 held out.

**Architecture:** Preserve notebook cells 0-13 and replace the remaining cells with a shared seven-network workflow. Each subject is standardized from training data only; subject and episode arrays are concatenated with `lengths`, so one HMM supplies common state definitions without cross-sequence transitions.

**Tech Stack:** Python 3.11, NumPy, pandas, h5py, scikit-learn, hmmlearn 0.3.3, matplotlib, joblib, Jupyter notebook JSON.

## Global Constraints

- Modify `src/notebook/get_data_one_episode.ipynb` only below its existing initial S01E01 plots.
- Do not average subjects and do not fit separate subject HMMs.
- Train candidates on Seasons 1-4, select K on Season 5, refit on Seasons 1-5, and decode S01E01 last.
- Emotion labels may enter only after S01E01 state decoding.
- Use K values 4-12, five restarts, full covariance, 200 maximum iterations, and tolerance `1e-2`.
- Prefer short commented notebook cells and concise runtime errors over repeated assertions.
- Keep unrelated working-tree changes untouched.

---

### Task 1: Add a focused notebook contract test

**Files:**
- Create: `tests/test_shared_hmm_notebook.py`
- Test: `tests/test_shared_hmm_notebook.py`

**Interfaces:**
- Consumes: notebook JSON at `src/notebook/get_data_one_episode.ipynb`.
- Produces: static checks for compilable cells, the shared-only model structure, held-out target ordering, and a synthetic helper smoke test.

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

import numpy as np


NOTEBOOK = Path("src/notebook/get_data_one_episode.ipynb")


def code_source():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_shared_hmm_notebook_contract():
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), NOTEBOOK.name, "exec")

    source = code_source()
    assert "## Shared concatenated HMM" in "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    assert "np.concatenate(transformed_sequences" in source
    assert "lengths" in source
    assert "validation_mean" in source
    assert "final_model.predict_proba" in source
    assert "X_subjects.mean(axis=0)" not in source
    assert "for subject in SUBJECTS:\n    for network" not in source


def test_sequence_scoring_preserves_boundaries():
    notebook = json.loads(NOTEBOOK.read_text())
    helper = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def score_sequences" in "".join(cell["source"])
    )

    namespace = {"np": np}
    exec(helper, namespace)

    class LengthModel:
        def score(self, values):
            return float(len(values) * 2)

    values = np.zeros((7, 2))
    scores = namespace["score_sequences"](LengthModel(), values, [3, 4])
    assert scores.tolist() == [2.0, 2.0]
```

- [ ] **Step 2: Run the contract test and confirm it fails against the old notebook**

Run: `uv run --extra test pytest tests/test_shared_hmm_notebook.py -q`

Expected: at least one failure because the old notebook lacks the shared-only heading and concatenated helper.

### Task 2: Replace the HMM section with the shared workflow

**Files:**
- Modify: `src/notebook/get_data_one_episode.ipynb`, cells after index 13
- Test: `tests/test_shared_hmm_notebook.py`

**Interfaces:**
- Consumes: `PROJECT_ROOT`, `emo_file`, HDF5 fMRI files, `SUBJECTS`, `NETWORKS`, `SegmentKey`, `discover_fmri_segments`, `common_segments`, and `network_indices`.
- Produces: `selection_table`, `selected_k`, `final_model`, `decoded_by_subject`, and `state_emotion_table`.

- [ ] **Step 1: Add configuration and data helpers**

Add short cells defining output paths and these functions:

```python
def load_network_sequences(h5_path, discovered, segments):
    sequences, ids = [], []
    with h5py.File(h5_path, "r") as handle:
        for segment in segments:
            key = discovered.get(segment)
            if key is None:
                raise ValueError(f"Missing segment: {segment.task}")
            parcels = handle[key][:]
            values = np.column_stack([
                parcels[:, PARCEL_INDICES[network]].mean(axis=1)
                for network in NETWORKS
            ])
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values: {segment.task}")
            sequences.append(values)
            ids.append(segment.task)
    return sequences, ids


def build_group_data(segments, scalers=None):
    fit_scalers = scalers is None
    scalers = {} if fit_scalers else scalers
    transformed_sequences, lengths, keys = [], [], []
    for subject in SUBJECTS:
        sequences, ids = load_network_sequences(
            H5_PATHS[subject], DISCOVERED_BY_SUBJECT[subject], segments
        )
        if fit_scalers:
            scalers[subject] = StandardScaler().fit(np.concatenate(sequences))
        for sequence, segment_id in zip(sequences, ids):
            transformed = scalers[subject].transform(sequence)
            transformed_sequences.append(transformed)
            lengths.append(len(transformed))
            keys.append((subject, segment_id))
    values = np.concatenate(transformed_sequences)
    if sum(lengths) != len(values):
        raise ValueError("Sequence lengths do not match concatenated data")
    return values, lengths, keys, scalers
```

- [ ] **Step 2: Add minimal HMM and summary helpers**

Implement `fit_best_hmm(k, values, lengths)`, `score_sequences(model, values, lengths)`, and `summarize_state_emotion(...)`. The fit helper tries five deterministic seeds, ignores only expected numerical fit exceptions, keeps the highest finite training LL, and raises one error if every restart fails. `score_sequences` splits using `lengths` and returns log-likelihood per TR.

- [ ] **Step 3: Add held-out segment discovery and candidate selection**

Build common train, validation, final-train, and target segment tuples. Fit each K once through `fit_best_hmm`, record `train_ll`, `aic`, `bic`, validation mean LL per TR, validation standard error, and iterations, then select the highest validation mean with smaller K as the tie-breaker.

- [ ] **Step 4: Add the model-selection plot**

Create a two-panel figure. The left panel plots AIC/BIC and training LL on a secondary axis. The right panel plots Season 5 LL per TR with standard-error bars and a vertical line at `selected_k`. Save `model_selection.png` and display it.

- [ ] **Step 5: Add final fitting and target decoding**

Fit new subject scalers and one final shared HMM on Seasons 1-5 excluding S01E01. Transform and decode the two S01E01 parts for each subject, then split `gamma` and Viterbi states back into a dictionary keyed by subject. Save the model, scalers, network order, target keys, and selected K in `shared_hmm.joblib`.

- [ ] **Step 6: Add emotion summaries and diagnostic plots**

Create `state_emotion_by_subject.csv`, one state-by-network profile heatmap, one four-panel target timeline per subject, and one two-panel subject-by-state consistency heatmap for high-minus-low arousal and positive-minus-negative valence.

- [ ] **Step 7: Run the focused test**

Run: `uv run --extra test pytest tests/test_shared_hmm_notebook.py -q`

Expected: `2 passed`.

### Task 3: Verify the notebook handoff

**Files:**
- Verify: `src/notebook/get_data_one_episode.ipynb`
- Verify: `tests/test_shared_hmm_notebook.py`

**Interfaces:**
- Consumes: completed notebook and focused test.
- Produces: evidence that the notebook is structurally runnable without claiming the expensive full fit was executed.

- [ ] **Step 1: Parse and compile every code cell**

Run:

```bash
uv run python -c 'import json; from pathlib import Path; p=Path("src/notebook/get_data_one_episode.ipynb"); n=json.loads(p.read_text()); [compile("".join(c["source"]), p.name, "exec") for c in n["cells"] if c["cell_type"]=="code"]; print("compiled", sum(c["cell_type"]=="code" for c in n["cells"]), "code cells")'
```

Expected: prints the number of compiled cells with exit code 0.

- [ ] **Step 2: Confirm the diff is scoped**

Run: `git status --short && git diff --stat && git diff -- tests/test_shared_hmm_notebook.py`

Expected: the notebook HMM section, focused test, and planning documents are the only task-related changes; unrelated existing files remain unstaged.

- [ ] **Step 3: Commit task files only**

```bash
git add -f src/notebook/get_data_one_episode.ipynb tests/test_shared_hmm_notebook.py
git commit -m "feat: simplify shared HMM notebook"
```

Do not stage `.gitignore`, data, outputs, editor settings, or other existing untracked files.
