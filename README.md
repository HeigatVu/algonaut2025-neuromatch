# algonauts-brainstates: Dynamic Brain State & Emotion Modeling

This repository provides tools and analysis pipelines for decoding dynamic brain states and emotional dynamics during naturalistic video watching (Friends TV series) using functional magnetic resonance imaging (fMRI) and Hidden Markov Models (HMM).

This project was developed for the Algonauts Project / Neuromatch challenge.

---

## Overview

Naturalistic movie watching evokes complex, continuous reconfigurations of whole-brain functional networks. This project investigates:
1. **Latent Brain State Discovery**: Unsupervised discovery of recurring cortical network states using Gaussian Hidden Markov Models (GaussianHMM).
2. **Multi-Subject Shared & Network-Specific Dynamics**: Fitting shared concatenated HMMs across subjects (`sub-01`, `sub-02`, `sub-03`, `sub-05`) as well as individual 7-network decompositions.
3. **Multimodal Emotion Alignment**: Mapping discrete emotions (joy, anger, sadness, surprise, neutral) and sentiment from the Multimodal EmotionLines Dataset (MELD) to fMRI time series with hemodynamic lag adjustments.
4. **Cortical Back-Projection**: Projecting latent state profiles and emotion-related condition differences back onto the 1000-parcel Schaefer 2018 cortical surface atlas.

---

## Key Features

- **Automated Data Discovery & Auditing**:
  - Discovers H5 segment keys across fMRI sessions and verifies 287 common segments across all 4 subjects.
  - Validates dimensionality, finite values, and TR lengths (`TR = 1.49s`).
  - Includes conservative artifact guards for global signal outliers.
- **Schaefer 1000-Parcel / 7 Yeo Network Mapping**:
  - Automatically fetches and parses the Schaefer 2018 1000-ROI atlas.
  - Maps cortical parcels to the 7 canonical Yeo functional networks: Visual (`VIS`), Somatomotor (`SMN`), Dorsal Attention (`DAN`), Salience / Ventral Attention (`SAL`), Limbic (`LIM`), Frontoparietal Control (`FPC`), and Default Mode (`DN`).
- **Robust Parallel HMM Training & Model Selection**:
  - Multi-restart Gaussian HMM fitting parallelized across CPU cores with `joblib`.
  - Candidate model selection across state counts $K$ using cross-validated log-likelihood and the one standard error (1-SE) heuristic.
  - Model health criteria enforcing minimum state occupancy (`min_state_trs`) and bounded emission means (`max_abs_mean`).
- **MELD Emotion Dataset Alignment**:
  - Automated downloading and caching of MELD transcript annotations.
  - Timestamp parsing (`start_s`, `end_s`) and split alignment with Algonauts episode splits (`a` and `b`).
  - Hemodynamic lag compensation (`PRIMARY_LAG = 3` TRs ~ 4.47s).
  - Pearson correlation and posterior-weighted contrast mapping for emotion categories and valence/sentiment contrasts.

---

## Project Structure

```text
Neuromatch-Algonauts2026/
├── pyproject.toml              # Build config and dependencies
├── src/
│   ├── preprocessing/
│   │   ├── brainstates/        # Core fMRI and HMM processing modules
│   │   │   ├── __init__.py
│   │   │   ├── config.py       # Subjects, networks, TR, and path settings
│   │   │   ├── data.py         # H5 segment discovery, parsing, validation
│   │   │   ├── networks.py     # Schaefer 1000 atlas and 7-network parcel mapping
│   │   │   └── audit.py        # Dataset-wide segment and temporal auditing
│   │   └── meld-dataset/       # Emotion dataset processing
│   │       └── meld.py         # MELD download, timestamp parsing, and split assignment
│   └── notebook/
│       ├── get_data_one_episode.ipynb               # Shared HMM model selection and fitting
│       ├── all_subjects_s03e15_network_emotion.ipynb # Multi-subject S03E15 network HMM emotion analysis
│       ├── sub01_s01e01_network_emotion.ipynb       # Single-subject S01E01 pilot workflow
│       └── plot.ipynb                               # Visualization and cortical map plotting
├── tests/                      # Automated test suite
│   ├── test_hmm_training_parallelism.py
│   └── test_s03e15_shared_network_hmm.py
└── outputs/                    # Exported models, figures, correlation tables, and audit logs
```

---

## Installation

### Requirements
- Python >= 3.11
- Package manager: [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### Setup with `uv`

```bash
# Clone the repository
git clone https://github.com/HeigatVu/algonaut2025-neuromatch.git
cd algonaut2025-neuromatch

# Create virtual environment and install dependencies
uv sync --extra test
```

### Setup with standard `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

---

## Data Configuration

The project expects fMRI BOLD data formatted in HDF5 (`.h5`) under `data/algonauts_2025.competitors/fmri/` or as configured in `src/preprocessing/brainstates/config.py`:

```python
SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05")
NETWORKS = ("VIS", "SMN", "DAN", "SAL", "LIM", "FPC", "DN")
TR_SECONDS = 1.49
```

MELD annotations are automatically retrieved from GitHub and cached in `data/cache` or local directories when calling `load_meld()`.

---

## Running Tests

All unit tests and notebook contracts can be verified with pytest:

```bash
# Using uv
uv run python -m pytest

# Using standard virtual environment
pytest
```

---

## Analysis Pipelines & Notebooks

1. **Shared Concatenated HMM (`src/notebook/get_data_one_episode.ipynb`)**:
   - Concatenates multi-subject time series across 7 networks.
   - Evaluates candidate models for $K \in [4, 17]$ with 20 parallel workers and 1-SE selection.
   - Decodes state posterior probabilities ($\gamma$) and state occupancy timelines.

2. **Network-Specific Emotion Decoding (`src/notebook/all_subjects_s03e15_network_emotion.ipynb`)**:
   - Performs PCA (e.g., 10 components) per functional network.
   - Trains 4-state Gaussian HMMs per network with full covariance.
   - Computes Pearson correlations between state posteriors and MELD emotion categories with hemodynamic shift.
   - Back-projects state-specific BOLD activation patterns to all 1000 cortical parcels.

3. **Single-Subject Pilot (`src/notebook/sub01_s01e01_network_emotion.ipynb`)**:
   - Computes direct BOLD vs. HMM-implied cortical contrast maps for valence and arousal.

---

## License

This project is licensed under the terms of the project repository.
