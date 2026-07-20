# Shared Concatenated HMM Notebook Design

## Goal

Replace only the HMM section of `src/notebook/get_data_one_episode.ipynb`
with a shorter, commented workflow that learns one shared HMM from all four
subjects while keeping `S01E01` fully held out until final decoding.

The model must not average subjects and must not fit separate subject HMMs.
It will concatenate subject sequences and pass their individual sequence
lengths to `GaussianHMM`, giving every subject the same state definitions
without inventing transitions between subjects or episodes.

## Scope

Keep the notebook's existing path setup, S01E01 loading, and initial plots.
Replace the current held-out HMM section with small markdown and code chunks.
The new section will:

1. Create seven network-mean features for each subject and episode.
2. Standardize each subject using training data only.
3. Train shared candidate HMMs on Seasons 1-4, excluding S01E01.
4. Select the state count using Season 5 likelihood per TR.
5. Refit one shared HMM on Seasons 1-5, still excluding S01E01.
6. Decode S01E01 for each subject with the shared model.
7. Relate decoded states to valence and arousal only after decoding.
8. Save compact tables, the final model, and diagnostic figures.

Non-goals are subject averaging, separate subject HMMs, supervised emotion
classification, formal population inference, and per-candidate cache logic.

## Data Flow

The seven features are parcel means within the Schaefer networks. For every
subject, each episode part remains a separate sequence.

Candidate selection uses this split:

- Training: common Seasons 1-4 segments except S01E01.
- Validation: common Season 5 segments.
- Target: S01E01 parts A and B, unused during fitting and K selection.

A `StandardScaler` is fitted separately for each subject on that subject's
training rows. The scaler transforms the same subject's training and
validation sequences. All transformed sequences are concatenated for the
shared HMM, and their original lengths are passed to `fit` and `score`.

After K selection, new subject-specific scalers are fitted on Seasons 1-5
except S01E01. The final shared HMM is fitted on the concatenated transformed
sequences. S01E01 is then transformed and decoded. The combined state and
posterior arrays are split back into subjects using recorded boundaries.

Manual emotion labels do not influence scaling, K selection, HMM fitting, or
state decoding. They enter only when summarizing the decoded S01E01 results.

## Model Selection

Use deliberately small, visible configuration constants:

- Candidate states: 4 through 12.
- Restarts: 5 per state count.
- Covariance: full.
- Maximum EM iterations: 200.
- Convergence tolerance: `1e-2`.

For each state count, keep the valid restart with the highest training
log-likelihood. Record training LL, AIC, BIC, validation mean LL per TR, and
validation standard error. Select the state count with the highest validation
mean, breaking an exact tie in favor of fewer states. AIC and BIC remain
diagnostics and do not inspect S01E01.

## Notebook Structure

The replacement section will use short commented cells:

1. Imports, paths, and configuration.
2. Three small helpers: load network sequences, fit the best restart, and
   score independent sequences.
3. Discover common segments and define train, validation, and target splits.
4. Build candidate training and validation data.
5. Fit candidate state counts and create the selection table.
6. Plot LL, AIC, BIC, and held-out validation LL.
7. Rebuild Seasons 1-5 data and fit the final shared HMM.
8. Decode S01E01 and split results back by subject.
9. Summarize state-emotion relationships.
10. Plot and save diagnostics.

The notebook will avoid wrapper classes, configuration objects, repeated
integrity cells, and dozens of asserts. It will retain only checks that prevent
misaligned sequences, missing data, non-finite input, or using an invalid HMM.

## Outputs and Plots

Write results under `outputs/hmm_shared_s01e01`:

- `shared_hmm.joblib`: final scaler dictionary and shared HMM.
- `model_selection.csv`: metrics for each candidate state count.
- `state_emotion_by_subject.csv`: posterior and Viterbi occupancy by subject,
  state, label, and label value.
- `model_selection.png`: AIC/BIC and training LL beside Season 5 LL per TR.
- `state_network_profiles.png`: shared state means across seven networks.
- `subject_<id>_timeline.png`: posterior heatmap, Viterbi sequence, valence,
  and arousal for each subject.
- `emotion_state_consistency.png`: subject-by-state differences for high minus
  low arousal and positive minus negative valence.

Posterior means are the primary state-emotion summary because they preserve
uncertainty. Viterbi occupancy is included as a simpler supporting measure.

## Error Handling and Verification

Raise concise errors when required segments are missing, arrays contain
non-finite values, emotion labels do not align with target TRs, or every HMM
restart fails. Keep one length-consistency check at concatenation boundaries.

Verification will parse the notebook as valid JSON, compile every Python code
cell, and run a small synthetic smoke check of the shared-HMM helpers. The full
multi-season fit will not be run during editing because it is computationally
expensive; the notebook will expose progress per candidate state count for the
user's full run.
