# HMM Training Parallelism Design

## Goal

Limit HMM training in `src/notebook/get_data_one_episode.ipynb` to 20 CPU
workers while substantially reducing model-selection runtime for candidate
state counts 4 through 17.

## Scope

Only HMM fitting will be CPU-limited and parallelized. Data loading, feature
construction, scoring, plotting, output paths, and saved result formats remain
unchanged.

The notebook will continue to use:

- Five deterministic restarts per state count.
- Full covariance matrices.
- At most 200 EM iterations with tolerance `1e-2`.
- Highest finite training likelihood to choose the restart for each state
  count.
- Season 5 validation likelihood to select the final state count.

## Training Design

Set `K_VALUES` to 4 through 17 and add a visible `N_JOBS = 20` configuration
constant. Treat every `(state count, restart)` pair as one independent job,
giving 70 jobs in total.

Use the already installed `joblib` process backend with at most 20 workers.
Set each worker's inner numerical thread limit to one so 20 concurrent fits do
not each create additional BLAS or OpenMP threads. Joblib will memory-map the
large read-only NumPy inputs instead of requiring a new project dependency or
custom shared-memory code.

Each job returns its state count, restart, fitted model, and finite training
log-likelihood. Expected numerical fit failures remain local to that job. After
all jobs finish, group results by state count and keep the model with the
highest likelihood. Raise the existing concise error when every restart for a
state count fails.

The same helper will fit the final selected state count. That call has only
five restart jobs, so it will use no more than five workers while retaining the
20-worker ceiling.

## Progress and Behavior

Enable joblib's built-in progress messages so the notebook reports completed
restart jobs and elapsed time while model selection runs. Existing per-state
selection metrics remain unchanged after aggregation.

Parallel task completion order may vary, but deterministic restart seeds and
likelihood-based aggregation preserve the model-selection rule. Exact floating
point results can still vary across numerical library and hardware versions,
as they can in the serial implementation.

## Verification

Verification will:

1. Parse the notebook JSON and compile every code cell.
2. Run the focused notebook tests.
3. Run a small synthetic helper check that confirms parallel restart results
   are grouped by state count and the best finite restart is retained.
4. Inspect the final diff to confirm only the notebook, its focused test, and
   this approved documentation are touched.

The full 70-fit dataset run is intentionally excluded from edit-time checks
because it is the expensive workload this change is designed to accelerate.
