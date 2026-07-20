# Robust HMM Data Quality and Selection Design

## Goal

Prevent known fMRI artifacts from creating degenerate HMM states, keep the
source HDF5 files unchanged, and select the simplest healthy state count that
is statistically competitive on held-out data.

## Scope

Modify the shared-HMM section of
`src/notebook/get_data_one_episode.ipynb`. Preserve the seven network-mean
features, subject-specific scaling, Seasons 1-4 training split, Season 5
validation split, held-out S01E01 target, five deterministic restarts, full
covariance, and 20-worker training limit.

Emotion association remains a later analysis phase. PCA, source-data edits,
automatic interpolation, and broad preprocessing changes are out of scope.

## Known Artifact Handling

The data-quality audit identified one affected subject-segment:

- Subject: `sub-02`
- Segment: `s05e12b`
- Extreme global TRs: 7, 11, 16, and 19

These TRs exceed 20 robust median absolute deviations in all seven network
means. No other subject-segment in Seasons 1-5 meets that conservative
criterion.

Add a visible `EXCLUDED_SEGMENTS_BY_SUBJECT` configuration mapping and omit
only `sub-02/s05e12b` in memory. Do not modify the HDF5 file. This removes
592 rows from approximately 446,000 final-training rows and one sequence from
Season 5 validation.

The exclusion must apply whenever group data is built, including candidate
validation and final training. It must not affect S01E01 or other subjects'
copies of S05E12b.

## Data-Quality Guard

After computing seven network means for every non-excluded sequence, calculate
a robust z-score per network:

`abs(value - median) / (1.4826 * MAD)`

Flag a TR only when all seven network scores exceed 20. If any non-excluded
sequence contains a flagged TR, raise an error that names the subject file,
segment, and TR indices. This is a guard against silently training on another
artifact, not an automatic cleaning mechanism.

A zero MAD cannot establish an extreme deviation for that network and must not
cause division by zero.

## Model Health Diagnostics

For the best restart at each K, calculate posterior state occupancy on the
training matrix and record:

- `min_state_trs`: the smallest expected number of training TRs assigned to
  any state.
- `max_abs_mean`: the largest absolute standardized emission mean.
- `healthy`: true when `min_state_trs >= 50` and
  `max_abs_mean <= 10`.

These conservative thresholds reject singleton-like states and emission means
that are implausibly extreme after standardization. Save the fields beside the
existing likelihood, AIC, BIC, convergence, and validation metrics.

If every candidate is unhealthy, stop with a concise error rather than fitting
a final model.

## K Selection

Keep held-out Season 5 log-likelihood per TR as the primary selection
criterion. Among healthy candidates:

1. Find the K with the highest validation mean.
2. Calculate a cutoff as that mean minus its validation standard error.
3. Select the smallest K whose validation mean is at or above the cutoff.

AIC and BIC remain diagnostics. They are not required to reach an interior
minimum in a finite candidate range. Keep the initial K range at 4 through 17;
extend it only after a clean run if the healthy validation curve still gives no
stable simpler solution.

## Outputs

Write the new run under `outputs/hmm_shared_s01e01_clean` so the current
artifact-contaminated outputs remain available for comparison.

Retain the existing model, tables, and figures. The model-selection table and
plot will additionally expose model-health metrics and identify the
one-standard-error selected K.

The saved model metadata will include the exclusion mapping and health
thresholds so the training conditions are recoverable.

## Error Handling

Raise concise errors when:

- A non-excluded sequence contains a global artifact.
- Every restart fails for a candidate K.
- Every candidate K is unhealthy.
- Target segments or emotion labels are missing or misaligned.

Do not silently clip, interpolate, winsorize, or modify source data.

## Verification

Focused tests will verify:

1. The known exclusion affects only `sub-02/s05e12b`.
2. A synthetic global spike is detected while ordinary data passes unchanged.
3. Zero-MAD features do not divide by zero or create false flags.
4. The one-standard-error rule selects the smallest eligible healthy K.
5. Unhealthy candidates cannot be selected.
6. Every notebook code cell compiles.

A lightweight synthetic HMM smoke test may run, but the full multi-season fit
will remain a user-run verification because it is computationally expensive.

