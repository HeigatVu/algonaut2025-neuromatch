import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


NOTEBOOK = Path("notebooks/get_data_one_episode.ipynb")


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
    assert "TARGET_LABEL_LENGTHS =" in config
    assert "candidate_fits = fit_best_hmms(" in selection
    assert "K_VALUES, train_values, train_lengths" in selection
    assert "min_state_trs" in selection
    assert "max_abs_mean" in selection
    assert "healthy" in selection
    assert "select_k_one_se(selection_table)" in selection
    assert "final_model, final_train_ll = fit_best_hmms(" in final_fit
    assert "(selected_k,), final_train_values" in final_fit
    assert "if not final_healthy:" in final_fit


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
    constant[10] = 100
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
        arrays = [
            np.arange(14, dtype=float).reshape(2, 7) + index
            for index, _ in enumerate(segments)
        ]
        return arrays, list(segments)

    namespace["load_network_sequences"] = fake_load
    values, lengths, keys, _ = namespace["build_group_data"](("good", "bad"))

    assert calls == {"one": ("good", "bad"), "two": ("good",)}
    assert lengths == [2, 2, 2]
    assert keys == [
        ("sub-01", "good"),
        ("sub-01", "bad"),
        ("sub-02", "good"),
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

    table["healthy"] = False
    with pytest.raises(RuntimeError, match="No healthy HMM candidates"):
        namespace["select_k_one_se"](table)


def test_selection_plot_exposes_model_health():
    plot = cell_source("shared-selection-plot")
    assert "min_state_trs" in plot
    assert "max_abs_mean" in plot
    assert "MIN_STATE_TRS" in plot
    assert "MAX_ABS_STATE_MEAN" in plot


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


def test_target_split_trims_unlabeled_tail_per_segment():
    namespace = {
        "np": np,
        "SUBJECTS": ("sub-01",),
        "target_keys": [("sub-01", "s01e01a"), ("sub-01", "s01e01b")],
        "target_lengths": [4, 4],
        "target_gamma": np.arange(16, dtype=float).reshape(8, 2),
        "target_states": np.arange(8),
        "TARGET_LABEL_LENGTHS": {"s01e01a": 4, "s01e01b": 2},
    }
    exec(cell_source("shared-split-target"), namespace)

    decoded = namespace["decoded_by_subject"]["sub-01"]
    assert decoded["gamma"].shape == (6, 2)
    assert decoded["states"].tolist() == [0, 1, 2, 3, 4, 5]
    assert decoded["part_lengths"] == [4, 2]


PILOT_NOTEBOOK = Path("notebooks/sub01_s01e01_network_emotion.ipynb")


def pilot_notebook_cells():
    return json.loads(PILOT_NOTEBOOK.read_text())["cells"]


def pilot_cell_source(cell_id):
    return next(
        "".join(cell["source"])
        for cell in pilot_notebook_cells()
        if cell.get("id") == cell_id
    )


def test_pilot_notebook_has_the_minimal_heldout_contract():
    for cell in pilot_notebook_cells():
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), PILOT_NOTEBOOK.name, "exec")

    source = "\n".join(
        "".join(cell["source"])
        for cell in pilot_notebook_cells()
        if cell["cell_type"] == "code"
    )
    assert "SUBJECT = 'sub-01'" in source
    assert "K_VALUES = tuple(range(1, 7))" in source
    assert "N_JOBS = 20" in source
    assert "PRIMARY_LAG = 3" in source
    assert "SegmentKey(1, 1, 'a')" in source
    assert "SegmentKey(1, 1, 'b')" in source
    assert "segment not in TARGET_SEGMENTS" in source
    assert "outputs/sub01_s01e01_network_emotion" in source
    assert "n_jobs=min(N_JOBS, len(tasks))" in source
    assert "inner_max_num_threads=1" in source
    assert "brain_contrast_maps.csv" in source
    assert "direct_bold_valence.png" in source
    assert "direct_bold_arousal.png" in source
    assert "hmm_implied_valence.png" in source
    assert "hmm_implied_arousal.png" in source
    assert "meld" not in source.lower()


def test_pilot_label_shift_stays_within_each_segment():
    namespace = {"pd": pd}
    exec(pilot_cell_source("pilot-label-helpers"), namespace)

    first = pd.DataFrame({"valence": [1, 0, -1], "arousal": [1, 0, 0]})
    second = pd.DataFrame({"valence": [-1, 0, 1], "arousal": [0, 0, 1]})
    shifted = pd.concat(
        [namespace["shift_labels"](frame, 1) for frame in (first, second)],
        ignore_index=True,
    )

    pd.testing.assert_series_equal(
        shifted["valence"],
        pd.Series([np.nan, 1, 0, np.nan, -1, 0], name="valence"),
    )
    pd.testing.assert_series_equal(
        shifted["arousal"],
        pd.Series([np.nan, 1, 0, np.nan, 0, 0], name="arousal"),
    )


def test_pilot_total_variation_compares_mean_posterior_vectors():
    namespace = {"np": np}
    exec(pilot_cell_source("pilot-association-helpers"), namespace)

    gamma = np.array([[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]])
    labels = np.array([1, 1, -1, -1], dtype=float)

    assert np.isclose(namespace["total_variation"](gamma, labels, 1, -1), 0.8)


def test_pilot_condition_difference_preserves_contrast_direction():
    namespace = {"np": np}
    exec(pilot_cell_source("pilot-brain-helpers"), namespace)

    values = np.array([[3.0, 2.0], [5.0, 4.0], [1.0, 6.0], [3.0, 4.0]])
    labels = np.array([1, 1, -1, -1], dtype=float)

    np.testing.assert_allclose(
        namespace["condition_mean_difference"](values, labels, 1, -1),
        [2.0, -2.0],
    )


def test_pilot_hmm_contrast_weights_back_projected_state_means():
    namespace = {"np": np}
    exec(pilot_cell_source("pilot-brain-helpers"), namespace)

    state_parcel_means = np.array([[1.0, 0.0], [0.0, 2.0]])
    gamma = np.array([[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]])
    labels = np.array([1, 1, -1, -1], dtype=float)

    np.testing.assert_allclose(
        namespace["posterior_weighted_parcel_contrast"](
            state_parcel_means, gamma, labels, 1, -1
        ),
        [0.8, -1.6],
    )


def test_pilot_parcel_values_project_to_atlas_labels():
    namespace = {"np": np}
    exec(pilot_cell_source("pilot-brain-helpers"), namespace)

    volume = namespace["parcel_values_to_volume"](
        np.array([0.5, -0.25, 1.0]),
        np.array([[0, 1], [3, 2]]),
    )

    np.testing.assert_allclose(volume, [[0.0, 0.5], [1.0, -0.25]])


def test_pilot_one_standard_error_rule_uses_only_healthy_candidates():
    namespace = {"np": np}
    exec(pilot_cell_source("pilot-model-helpers"), namespace)

    table = pd.DataFrame({
        "k": [1, 2, 3],
        "validation_mean": [0.89, 0.92, 1.00],
        "validation_se": [0.05, 0.05, 0.12],
        "healthy": [True, True, True],
    })
    assert namespace["select_k_one_se"](table) == (1, 0.88)

    table.loc[table["k"].eq(1), "healthy"] = False
    assert namespace["select_k_one_se"](table) == (2, 0.88)


def test_pilot_hmm_restart_helper_runs_on_small_real_data():
    import warnings

    from hmmlearn.hmm import GaussianHMM
    from joblib import Parallel, delayed, parallel_config

    namespace = {
        "np": np,
        "warnings": warnings,
        "GaussianHMM": GaussianHMM,
        "Parallel": Parallel,
        "delayed": delayed,
        "parallel_config": parallel_config,
        "N_JOBS": 1,
        "SEEDS": (2025, 2026),
        "HMM_N_ITER": 20,
        "HMM_TOL": 1e-3,
        "MIN_STATE_OCCUPANCY": 0.01,
    }
    exec(pilot_cell_source("pilot-model-helpers"), namespace)
    values = np.random.default_rng(0).normal(size=(40, 2))

    model, score, occupancy, seed = namespace["fit_best_hmms"](
        (1,), values, [20, 20]
    )[1]

    assert model.n_components == 1
    assert np.isfinite(score)
    assert occupancy == 1.0
    assert seed in (2025, 2026)
