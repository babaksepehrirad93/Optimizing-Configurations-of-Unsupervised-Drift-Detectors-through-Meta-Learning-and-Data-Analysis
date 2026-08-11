# Metadata-Driven Recommendation of Hyperparameter Configurations for Unsupervised Concept Drift Detectors

This repository contains the framework developed by **Babak Sepehri Rad** for the master's thesis **“Optimizing Configurations of Unsupervised Drift Detectors through Meta-learning and Data Analysis”** at **TU Dresden**.

The thesis investigates whether results from previous concept-drift detector evaluations can be transformed into reusable meta-learning knowledge for recommending promising hyperparameter configurations on unseen datasets. The framework learns detector-specific relationships between dataset metadata, detector configurations, predictive accuracy, and runtime. It then predicts and refines candidate configurations without evaluating them in the target drift-detection pipeline during recommendation.

The framework recommends configurations; it does **not** execute the concept-drift detectors itself. Recommended configurations are evaluated with the benchmark pipeline of Werner et al.:

<https://github.com/ScaDS/benchmark-unsupervised-concept-drift-detection/tree/main>

Measured Phase 2 results obtained after executing the recommendations in that benchmark pipeline are provided under `results/phase_2/`. Phase 1 result material for **60 detector-dataset cases** is provided under `results/phase_1/`.

## Framework overview

The implementation follows two main phases:

1. **Phase 1 — Methodology selection**
   - preprocess existing benchmark results;
   - extract and prune dataset meta-features;
   - construct leave-one-dataset-out (LODO) metadata representations;
   - train detector-specific Extra Trees regression models;
   - compare scalar-target formulations with separate accuracy/runtime prediction;
   - evaluate predicted recommendations on held-out benchmark configurations.

2. **Phase 2 — Configuration recommendation and practical evaluation**
   - generate candidate configurations with Sobol sampling;
   - select prediction-based seeds;
   - refine candidates with TPE and, for Separate prediction, an independent NSGA-II branch;
   - select a fixed-size recommendation set from predicted Pareto layers;
   - execute the recommended configurations externally in the benchmark drift-detection pipeline;
   - evaluate the measured recommendations against benchmark-reference and default configurations.

Raw accuracy is treated as higher-is-better and raw runtime as lower-is-better. During learning, runtime is transformed so that both transformed objectives are higher-is-better.

## Main features

- Detector-specific LODO meta-learning across the configured datasets.
- Dataset metadata extraction from the initial stream prefix, including statistical, information-based, landmarking, tree-based, and early-shift descriptors.
- Metadata cleaning, correlation pruning, and train-only LODO PCA variants.
- Configuration-only (`CFG`), pruned metadata, LODO PCA, and experimental LODO PCA-ranked model inputs.
- Scalar learning targets: modified distance, Tchebycheff, PBI, APD, and additional experimental targets.
- Separate regression of transformed accuracy and transformed runtime.
- Extra Trees regression with shared, centrally configured model parameters.
- Prediction-only Phase 1 selection and evaluation against observed held-out benchmark results.
- Sobol candidate generation with TPE refinement and an independent NSGA-II branch for Separate recommendation.
- Static one-stage and dynamic multi-stage TPE refinement modes.
- Deterministic exact-budget Pareto-layer selection.
- Phase 1 and Phase 2 evaluation metrics, plots, and aggregated result exports.
- A pipeline runner for normal single-setup execution and Cartesian target × metadata sweeps.

## Repository structure

```text
.
├── preprocessing_data.py       # Benchmark preprocessing
├── metadata_extraction.py      # Dataset metadata extraction and PCA variants
├── train_models.py             # Detector-specific LODO model training
├── phase1_evaluation.py        # Held-out Phase 1 evaluation
├── config_recommendation.py    # Prediction-guided configuration recommendation
├── phase2_evaluation.py        # Evaluation of externally measured Phase 2 results
├── pipeline_runner.py          # Sequential execution / experiment sweeps
├── src/
│   ├── config.py               # User-adjustable pipeline configuration
│   ├── sweeper_setup.py        # Internal sweep setup interface
│   ├── training_data.py        # LODO dataset construction
│   ├── metadata_core.py        # Meta-feature computation and PCA
│   ├── target_utils.py         # Target formulations and Pareto utilities
│   ├── evaluation.py           # Phase 1 evaluation logic
│   ├── metrics.py              # Evaluation metrics
│   ├── selection.py            # Exact-budget Pareto selection
│   ├── model_factory.py        # Extra Trees model construction
│   ├── transforms.py           # Accuracy/runtime transformations
│   ├── plotting.py             # Phase 1 plots
│   ├── paths.py                # Centralized repository paths
│   └── utils.py                # Shared file/serialization helpers
├── data/
│   ├── raw_data/
│   │   ├── benchmarking_results/
│   │   ├── datasets/
│   │   └── search_space/
│   ├── processed_benchmark_data/
│   ├── extracted_metadata/
└── results/
    ├── phase_1/
    │   ├── held-out_evaluation/
    │   └── system_recommendations/
    └── phase_2/
        ├── drift_detection_pipeline_results/
        └── computed_metrics/
        └── plots/
```

Directories are resolved from the repository root through `src/paths.py`; scripts do not depend on the current working directory for artifact locations.

## Requirements

The code requires **Python 3.10 or newer**. Install the Python dependencies from `requirements.txt`.

A minimal environment can be created with:

```bash
python -m venv .venv
```

Activate the environment and install the dependencies, for example:

```bash
pip install -r requirements.txt
```

## Input data

The main input locations are defined in `src/paths.py`.

### Benchmark results

Place raw benchmark result files at:

```text
data/raw_data/benchmarking_results/<detector>/<detector>_<dataset>.csv
```

The preprocessing stage expects benchmark rows containing at least `Status`, `ACCURACY`, and `RUNTIME` in addition to detector configuration columns.

### Raw datasets

Place dataset CSV files at:

```text
data/raw_data/datasets/<dataset>.csv
```

These files are used by `metadata_extraction.py`. Metadata is computed from the configured initial stream prefix (`METADATA_N_PREFIX` in `src/config.py`).

### Detector search spaces

Configuration recommendation requires manually maintained detector search-space files:

```text
data/raw_data/search_space/<detector>_search_space.csv
```

The recommendation stage reads these files but does not derive or modify detector search spaces from benchmark performance.

## Configuration

All normal user-adjustable settings are centralized in:

```text
src/config.py
```

The file is organized by pipeline stage and documents the supported values for configurable options. The main groups are:

- benchmark preprocessing and objective transformations;
- metadata extraction, cleaning, pruning, and PCA;
- model-input metadata representation;
- target-learning mode and scalar formulations;
- Extra Trees model parameters;
- Phase 1 recommendation budget, eligibility threshold, metrics, and plotting;
- Phase 2 recommendation budget and Sobol/Optuna settings;
- static and dynamic TPE refinement;
- NSGA-II refinement for Separate prediction;
- pipeline-runner stage and sweep settings.

### Target mode

The two primary learning modes are:

```python
TRAIN_TARGET_MODE = "single"    # one scalar target
TRAIN_TARGET_MODE = "separate"  # separate accuracy and runtime regressors
```

For `single`, select the formulation with `SINGLE_TARGET_FORMULATION`. The standard thesis formulations are `mod_dist`, `tchebycheff`, `pbi`, and `apd`.

### Metadata representation

```python
TRAIN_USE_METADATA = False              # configuration-only (CFG)
TRAIN_USE_METADATA = True
TRAIN_METADATA_VARIANT = "pruned"       # or lodo_pca / lodo_pca_ranked
PCA_LODO_VARIANCE = 0.85                # used with LODO PCA variants
```

The standard LODO PCA explained-variance targets are `0.80`, `0.85`, `0.90`, and `0.95`.

### Recommendation refinement

```python
CONFIG_RECOMMENDATION_MODE = "static"   # one TPE refinement stage
CONFIG_RECOMMENDATION_MODE = "dynamic"  # three TPE stages with seed reselection
```

Warm-start Sobol seed evaluations are additional to the configured TPE/NSGA-II refinement trial counts. Separate recommendation uses independent TPE and NSGA-II refinement branches before final predicted Pareto-layer selection.

## Running individual stages

The five main pipeline stages can be executed independently. Each uses the current settings in `src/config.py`.

### 1. Benchmark preprocessing

```bash
python preprocessing_data.py --detector ALL --dataset ALL
```

A single detector/dataset or comma-separated exact names can also be provided:

```bash
python preprocessing_data.py --detector BNDM --dataset Electricity
python preprocessing_data.py --detector BNDM,UDetect --dataset Electricity,PokerHand
```

Output:

```text
data/processed_benchmark_data/<detector>/<detector>_<dataset>_processed.csv
```

### 2. Metadata extraction

```bash
python metadata_extraction.py
```

This stage computes the configured metadata exports for the available raw datasets, including train-only LODO PCA variants.

Outputs are written under:

```text
data/extracted_metadata/
```

### 3. Phase 1 model training

```bash
python train_models.py --detector ALL --dataset ALL
```

The `--dataset` argument identifies the LODO held-out dataset(s). Models are saved under `models/` using setup-specific artifact paths.

### 4. Phase 1 evaluation

```bash
python phase1_evaluation.py --detector ALL --dataset ALL
```

Useful options:

```bash
python phase1_evaluation.py --detector BNDM --dataset Electricity --skip-plots
python phase1_evaluation.py --detector BNDM --dataset Electricity --plots-only
```

Phase 1 results are written under:

```text
results/phase_1/held-out_evaluation/
```

The repository provides Phase 1 result material for 60 detector-dataset cases.

### 5. Configuration recommendation

```bash
python config_recommendation.py --detector ALL --dataset ALL
```

or, for one pair:

```bash
python config_recommendation.py --detector BNDM --dataset Electricity
```

The script loads the setup-specific trained models, predicts candidate quality, performs the configured refinement procedure, and exports recommended configurations under:

```text
results/phase_1/system_recommendations/
```

This stage does **not** run the recommended configurations in a drift-detection pipeline.

## Using `pipeline_runner.py`

`pipeline_runner.py` executes selected stages in their fixed pipeline order:

```text
preprocessing → metadata extraction → training → Phase 1 evaluation → configuration recommendation
```

Enable or disable stages in `src/config.py`:

```python
PIPELINE_RUN_PREPROCESSING = False
PIPELINE_RUN_METADATA_EXTRACTION = False
PIPELINE_RUN_TRAINING = True
PIPELINE_RUN_PHASE1_EVALUATION = True
PIPELINE_RUN_CONFIG_RECOMMENDATION = True
```

The runner does not infer dependencies. Upstream stages can be disabled when the required artifacts already exist.

### Single mode

```python
PIPELINE_RUN_MODE = "single"
```

Single mode executes the enabled stages once and uses the ordinary target, metadata, PCA, and recommendation settings from `src/config.py`. It does not create a second set of setup parameters.

Run all configured detectors/datasets:

```bash
python pipeline_runner.py --detector ALL --dataset ALL
```

or restrict the scope:

```bash
python pipeline_runner.py --detector BNDM --dataset Electricity
```

### Sweep mode

```python
PIPELINE_RUN_MODE = "sweep"
```

Sweep mode builds the Cartesian product of:

```python
PIPELINE_SWEEP_TARGET_FORMULATIONS
PIPELINE_SWEEP_METADATA_SETUPS
```

The default standard sweep contains five target formulations and six metadata setups:

```text
Targets:   mod_dist, tchebycheff, pbi, apd, separate
Metadata:  CFG, Pruned, LODO PCA 80, 85, 90, 95
```

This produces **30 setups**. The lists can be shortened to run only a selected subset.

The generated setup is passed only to setup-aware child processes. Normal standalone execution remains controlled by `src/config.py`.

### Dry run

Before starting a long execution, inspect the planned commands with:

```bash
python pipeline_runner.py --dry-run --detector ALL --dataset ALL
```

The runner is fail-fast: execution stops if an enabled child stage returns an error.

## External Phase 2 execution

The recommendation framework predicts promising hyperparameter configurations, but the real concept-drift detector execution is performed with the benchmark implementation by Werner et al.:

<https://github.com/ScaDS/benchmark-unsupervised-concept-drift-detection/tree/main>

Recommended configurations from `results/phase_1/system_recommendations/` can be transferred to that benchmark pipeline for measurement under the intended experimental environment.

The measured thesis results are stored in:

```text
results/phase_2/drift_detection_pipeline_results/
```

Each detector directory contains detector-dataset result files of the form:

```text
<detector>_<dataset>_final_results.csv
```

Phase 2 evaluation expects rows identified as recommendation (`Rec`), benchmark reference (`Par`), or default (`Def`) and requires measured `Status`, `ACCURACY`, and `RUNTIME` values. Standardized prediction columns are used when available for prediction-versus-measurement correlations.

## Phase 2 evaluation

After the externally measured result files are available, run:

```bash
python phase2_evaluation.py
```

The script computes pair-level and aggregated Phase 2 metrics and writes them to:

```text
results/phase_2/computed_metrics/
```

Outputs include detector-level, dataset-level, global, and combined metric tables. The Phase 2 evaluation script is intentionally separate from `pipeline_runner.py` because it consumes externally measured detector results.


## Results

The repository result directories are organized as follows:

```text
results/
├── phase_1/
│   ├── held-out_evaluation/       # Phase 1 predictions, metrics, and plots
│   └── system_recommendations/    # Generated configuration recommendations
└── phase_2/
    ├── drift_detection_pipeline_results/      # Measured Rec/Par/Def results from benchmark execution
    └── computed_metrics/          # Phase 2 pair/aggregate evaluation outputs
    └── plots/          # Phase 2 pair/plotted results
```

The included Phase 2 results correspond to recommendations evaluated with the external benchmark pipeline. The included Phase 1 material covers 60 detector-dataset cases.

## Methodological notes

- **HPO and recommendation are distinct:** candidate configurations are ranked and refined using learned prediction models; the recommendation procedure does not measure candidates on the target drift-detection task.
- **LODO separation:** model training and train-only PCA construction exclude the held-out dataset.
- **Prediction versus measurement:** predicted Pareto layers are used for recommendation; real/observed objectives are used for evaluation.
- **Runtime direction:** raw runtime is lower-is-better, while transformed runtime is reversed to higher-is-better for learning and Pareto operations.
- **Observed references:** benchmark Pareto sets represent the available benchmark observations and should not be interpreted as guaranteed global optima.
- **Reproducibility:** random seeds and principal experimental settings are centralized in `src/config.py`.

## Attribution

**Developer:** Babak Sepehri Rad  
**Master's thesis:** *Optimizing Configurations of Unsupervised Drift Detectors through Meta-learning and Data Analysis*  
**Institution:** Dresden University of Technology

The framework was developed as the implementation component of the master's thesis. When reusing the implementation or derived results, please reference the thesis and this repository. The external drift-detection benchmark remains the work of Werner et al. and is available from the repository linked above.
