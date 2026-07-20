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
    assert "candidate_fits = fit_best_hmms(" in selection
    assert "K_VALUES, train_values, train_lengths" in selection
    assert "final_model, final_train_ll = fit_best_hmms(" in final_fit
    assert "(selected_k,), final_train_values" in final_fit


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
