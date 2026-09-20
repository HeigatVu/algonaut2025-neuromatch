import json
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK = Path("notebooks/all_subjects_s03e15_network_emotion.ipynb")


def notebook_cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def cell_source(cell_id):
    return next(
        "".join(cell["source"])
        for cell in notebook_cells()
        if cell.get("id") == cell_id
    )


def test_notebook_has_s03e15_shared_network_contract():
    for cell in notebook_cells():
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), NOTEBOOK.name, "exec")

    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook_cells()
        if cell["cell_type"] == "code"
    )
    assert "SUBJECTS = ('sub-01', 'sub-02', 'sub-03', 'sub-05')" in source
    assert "SegmentKey(3, 15, 'a')" in source
    assert "SegmentKey(3, 15, 'b')" in source
    assert "N_STATES = 4" in source
    assert "N_COMPONENTS = 10" in source
    assert "MATCH_THRESHOLD = 0.90" in source
    assert "PRIMARY_LAG = 3" in source
    assert "covariance_type='full'" in source
    assert "outputs/all_subjects_s03e15_network_emotion" in source
    assert "pearson_correlations.csv" in source
    assert "state_bold_maps.csv" in source
    assert "label_bold_contrasts.csv" in source


def test_labels_are_filtered_padded_and_shifted_within_one_segment():
    namespace = {
        "np": np,
        "pd": pd,
        "LABEL_COLUMNS": ("meld_emotion", "meld_sentiment"),
        "MATCH_THRESHOLD": 0.90,
        "PRIMARY_LAG": 1,
    }
    exec(cell_source("s03e15-label-helpers"), namespace)

    frame = pd.DataFrame({
        "meld_emotion": ["neutral", "anger", "joy"],
        "meld_sentiment": ["neutral", "negative", "positive"],
        "match_ratio": [1.0, 0.80, 0.90],
    })
    shifted = namespace["prepare_segment_labels"](frame, target_length=4)

    assert len(shifted) == 4
    assert pd.isna(shifted.loc[0, "meld_emotion"])
    assert shifted.loc[1, "meld_emotion"] == "neutral"
    assert pd.isna(shifted.loc[2, "meld_emotion"])
    assert shifted.loc[3, "meld_emotion"] == "joy"
    assert shifted["accepted"].tolist() == [False, True, False, True]


def test_pearson_helper_uses_only_finite_nonconstant_rows():
    namespace = {"np": np}
    exec(cell_source("s03e15-association-helpers"), namespace)

    values = np.array([0.0, 99.0, 1.0, 2.0, 3.0])
    labels = np.array([0.0, np.nan, 1.0, 2.0, 3.0])
    assert np.isclose(namespace["pearson_r"](values, labels), 1.0)
    assert np.isnan(namespace["pearson_r"](np.ones(4), np.arange(4)))


def test_state_means_are_back_projected_and_subject_averaged():
    namespace = {"np": np}
    exec(cell_source("s03e15-brain-helpers"), namespace)

    class FakePCA:
        def inverse_transform(self, values):
            return np.asarray(values) + 1.0

    class FakeScaler:
        def __init__(self, offset):
            self.offset = offset

        def inverse_transform(self, values):
            return np.asarray(values) + self.offset

    result = namespace["back_project_state_means"](
        np.array([[1.0, 2.0]]),
        FakePCA(),
        {"sub-01": FakeScaler(1.0), "sub-02": FakeScaler(3.0)},
    )

    np.testing.assert_allclose(result, [[4.0, 5.0]])
