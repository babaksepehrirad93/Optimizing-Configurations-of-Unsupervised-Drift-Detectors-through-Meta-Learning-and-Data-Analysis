"""User-adjustable settings for the thesis recommendation pipeline.

The five runnable pipeline stages can be executed independently:
`preprocessing_data.py`, `metadata_extraction.py`, `train_models.py`,
`phase1_evaluation.py`, and `config_recommendation.py`.

`pipeline_runner.py` executes enabled stages sequentially. In `single` mode it
runs once using this file directly. In `sweep` mode it runs every combination in
`PIPELINE_SWEEP_TARGET_FORMULATIONS x PIPELINE_SWEEP_METADATA_SETUPS`.

`phase2_evaluation.py` is separate from the runner because it evaluates
externally measured Phase 2 Rec/Par/Def result files.
"""

from __future__ import annotations

from typing import Any, Literal


# =============================================================================
# General settings / paths / random seed
# =============================================================================
# Shared random seed used by deterministic model training and recommendation.
GLOBAL_RANDOM_SEED: int = 42

# Detector names supported by the benchmark and recommendation pipeline.
ALL_DETECTORS: list[str] = [
    "BNDM", "CDBD", "CDLEEDS", "CSDDM", "D3", "DAWIDD", "DDAL", "EDFS", "HDDDM", "IBDD",
    "IKS", "NNDVI", "OCDD", "PCACD", "SlidShaps", "SPLL", "STUDD", "UCDD", "UDetect", "WindowKDE",
]

# Dataset names supported by the benchmark and LODO pipeline.
ALL_DATASETS: list[str] = [
    "Electricity",
    "ForestCovertype",
    "GasSensor",
    "NOAAWeather",
    "OutdoorObjects",
    "Ozone",
    "PokerHand",
    "RialtoBridgeTimelapse",
    "SensorStream",
]

TRANSFORM_IDENTITY: Literal["identity"] = "identity"
TRANSFORM_MINMAX: Literal["minmax"] = "minmax"
TRANSFORM_LOG1P_MINMAX: Literal["log1p_minmax"] = "log1p_minmax"


# =============================================================================
# Benchmark preprocessing
# =============================================================================
# Controls row filtering and objective transformations applied to raw benchmark
# results before target construction and model training.

# Benchmark status retained for model/evaluation data.
PREPROCESSING_STATUS_TO_KEEP: str = "Completed"

# Objective transformation for accuracy: identity | minmax.
PREPROCESSING_ACCURACY_MODE: str = TRANSFORM_MINMAX

# Runtime transformation: log1p_minmax | minmax. Transformed runtime is higher-is-better.
PREPROCESSING_RUNTIME_MODE: str = TRANSFORM_LOG1P_MINMAX

# Export transformed_accuracy and transformed_runtime alongside raw objectives.
PREPROCESSING_EXPORT_TRANSFORMED_OBJECTIVES: bool = True

# Add a stable row index when raw benchmark files do not provide one.
PREPROCESSING_ADD_INDEX_IF_MISSING: bool = True

# Clip large raw runtimes before transformation when enabled.
PREPROCESSING_RUNTIME_UPPER_CLIPPING: bool = False

# Runtime clipping method: iqr_upper | percentile_upper.
PREPROCESSING_RUNTIME_CLIP_METHOD: str = "percentile_upper"

# Upper runtime threshold for IQR-based clipping.
PREPROCESSING_RUNTIME_CLIP_IQR_MULTIPLIER: float = 1.8

# Upper runtime percentile for percentile-based clipping.
PREPROCESSING_RUNTIME_CLIP_PERCENTILE: float = 99.6


# =============================================================================
# Metadata extraction: setup
# =============================================================================
# Controls metadata export stages from raw dataset files. These exports are
# consumed later according to the model-input metadata variant.

# Recompute metadata from raw datasets instead of reusing exported files.
METADATA_FRESH_COMPUTATION: bool = True

# Export per-dataset row/column/class overview.
METADATA_EXPORT_DATASET_OVERVIEW: bool = True

# Export all computed metadata before cleaning.
METADATA_EXPORT_ALL_META_FEATURES: bool = True

# Export metadata after NaN, near-constant, and duplicate-column removal.
METADATA_EXPORT_CLEANED_META_FEATURES: bool = True

# Export metadata after correlation pruning.
METADATA_EXPORT_PRUNED_META_FEATURES: bool = True

# Export train-only LODO PCA and LODO PCA ranked variants.
METADATA_EXPORT_LODO_PCA_VARIANTS: bool = True


# =============================================================================
# Metadata extraction: feature computation
# =============================================================================
# Controls dataset-level meta-feature computation from each raw dataset prefix.

# Number of initial stream samples used for metadata extraction.
METADATA_N_PREFIX: int = 2000

# Include simple landmarking model scores as metadata features.
METADATA_ENABLE_LANDMARKING: bool = True

# Include decision-tree descriptor features.
METADATA_ENABLE_TREE_DESCRIPTORS: bool = True

# Suppress expected convergence/class-balance warnings during metadata extraction.
METADATA_QUIET_WARNINGS: bool = True

# Maximum cross-validation folds for landmarking models.
LANDMARK_MAX_FOLDS: int = 5

# Minimum total samples required before fitting landmarking models.
LANDMARK_MIN_SAMPLES_TOTAL: int = 30

# Minimum samples per class required for cross-validated landmarking.
LANDMARK_MIN_PER_CLASS_FOR_CV: int = 3

# Maximum fitting iterations for the logistic-regression landmark model.
LOGREG_MAX_ITER: int = 2000

# Histogram-bin count used for normalized entropy metadata features.
ENTROPY_BINS: int = 10

# Batch size used for early-stream distribution-shift metadata features.
EARLYSHIFT_BATCH_SIZE: int = 200


# =============================================================================
# Metadata extraction: cleaning and pruning
# =============================================================================
# Defines column-level metadata cleanup. Any metadata column containing NaN is
# removed before near-constant, duplicate, and correlation pruning.

# Variance threshold for near-constant metadata columns.
NEAR_CONST_EPS: float = 1e-3

# Maximum allowed Pearson/Spearman absolute correlation before pruning a column.
CORR_PRUNE_THRESHOLD: float = 0.90

# Explained-variance targets for LODO PCA metadata variants.
PCA_VARIANCES: list[float] = [0.80, 0.85, 0.90, 0.95]

# Upper bound on retained PCA components per LODO fold.
PCA_MAX_COMPONENTS: int = 10


# =============================================================================
# Model training: metadata variants and usage
# =============================================================================
# Selects the metadata representation included in model inputs. CFG-only input
# is represented by disabling metadata.

# Include metadata columns in model inputs; False gives configuration-only input.
TRAIN_USE_METADATA: bool = True

# Model-input metadata representation: pruned | lodo_pca | lodo_pca_ranked.
TRAIN_METADATA_VARIANT: str = "lodo_pca"

# Default LODO PCA variance for standalone and runner single-mode execution.
PCA_LODO_VARIANCE: float = 0.85

# Number of ranked PCA-derived metadata features retained per LODO fold.
PCA_RANKED_NUMBER_OF_FEATURES: int = 1

# Scale metadata columns before model fitting.
TRAIN_SCALE_METADATA: bool = False

# Metadata scaling method: identity | minmax.
TRAIN_METADATA_SCALE_METHOD: str = TRANSFORM_MINMAX

# Detector/dataset scope for training and Phase 1 evaluation.
TRAIN_DETECTORS: list[str] = ALL_DETECTORS.copy()
TRAIN_DATASETS: list[str] = ALL_DATASETS.copy()


# =============================================================================
# Model training: learning-target formulations
# =============================================================================
# Defines whether models learn one scalar target or the Separate two-regressor target.

# Target-learning mode: single | separate.
TRAIN_TARGET_MODE: str = "separate"

# Single-target formulation: mod_dist | tchebycheff | pbi | apd |
# euc_dist | pareto_score | pareto_loss | pareto_rank.
SINGLE_TARGET_FORMULATION: str = "tchebycheff"

# Apply five preference directions to regional Tch/PBI/APD single-target training.
TRAIN_USE_PREFERENCE_REGIONS: bool = True

# Single-target formulations that support preference-region training.
REGIONAL_TARGETS: set[str] = {"tchebycheff", "pbi", "apd"}

# Derived from the active target mode, formulation, and regional toggle.
REGIONAL_ACTIVE: bool = (
    TRAIN_USE_PREFERENCE_REGIONS
    and TRAIN_TARGET_MODE == "single"
    and SINGLE_TARGET_FORMULATION in REGIONAL_TARGETS
)

# Thesis single-target formulations used in the standard sweep.
FINAL_SINGLE_TARGET_FORMULATIONS: tuple[str, ...] = (
    "mod_dist",
    "tchebycheff",
    "pbi",
    "apd",
)

# Additional supported scalar targets retained for experiments.
LEGACY_SINGLE_TARGET_FORMULATIONS: tuple[str, ...] = (
    "euc_dist",
    "pareto_score",
    "pareto_loss",
    "pareto_rank",
)

# Complete set of selectable single-target formulations.
SUPPORTED_SINGLE_TARGET_FORMULATIONS: tuple[str, ...] = (
    FINAL_SINGLE_TARGET_FORMULATIONS + LEGACY_SINGLE_TARGET_FORMULATIONS
)

# Default neutral accuracy weight; runtime weight is 1.0 minus this value.
DEFAULT_ACCURACY_WEIGHT: float = 0.5

# Names of the five fixed accuracy/runtime preference regions.
TRAIN_PREFERENCE_REGION_NAMES: tuple[str, str, str, str, str] = (
    "WACC_0p1_WRT_0p9",
    "WACC_0p3_WRT_0p7",
    "WACC_0p5_WRT_0p5",
    "WACC_0p7_WRT_0p3",
    "WACC_0p9_WRT_0p1",
)

# Accuracy weights for each preference region; runtime weight is 1.0 - weight.
TRAIN_PREFERENCE_REGION_ACCURACY_WEIGHTS: dict[str, float] = {
    "WACC_0p1_WRT_0p9": 0.1,
    "WACC_0p3_WRT_0p7": 0.3,
    "WACC_0p5_WRT_0p5": 0.5,
    "WACC_0p7_WRT_0p3": 0.7,
    "WACC_0p9_WRT_0p1": 0.9,
}

# Ideal point in transformed objective space for scalarizations.
TRAIN_SCALARIZATION_IDEAL_POINT: tuple[float, float] = (1.0, 1.0)

# Penalty multiplier for perpendicular distance in PBI scalarization.
TRAIN_PBI_THETA: float = 5.0

# Angular penalty exponent for APD scalarization.
TRAIN_APD_ALPHA: float = 2.0

# APD evaluation ratio; 1.0 applies the full angular penalty.
TRAIN_APD_EVAL_RATIO: float = 1.0


# =============================================================================
# Model training: model family, parameters, and artifacts
# =============================================================================
# Defines the Extra Trees regressor and artifact-writing behavior.

# Model family; only Extra Trees is supported.
TRAIN_MODEL_FAMILY: str = "ET"

# Optional model-parameter overrides merged with Extra Trees defaults.
TRAIN_MODEL_PARAMS: dict[str, Any] = {}

# Random state passed into each fitted regressor.
TRAIN_RANDOM_STATE: int = 42

# Save processed LODO training snapshots used to fit models.
TRAIN_SAVE_TRAINING_DATA_SNAPSHOT: bool = True

# Save fitted model artifacts.
TRAIN_SAVE_MODELS: bool = True

# Allow existing setup-specific artifacts to be overwritten.
TRAIN_OVERWRITE_ARTIFACTS: bool = False

# Supported model families; retained as a validation constant.
SUPPORTED_MODEL_FAMILIES = ("ET",)

# Default Extra Trees parameters used unless TRAIN_MODEL_PARAMS overrides them.
DEFAULT_REGRESSOR_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "ET": {
        "n_estimators": 300,
        "max_depth": 24,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": False,
        "n_jobs": -1,
    },
}


# =============================================================================
# Phase 1 evaluation: setup
# =============================================================================
# Controls held-out Phase 1 recommendation selection and metric evaluation.

# Number of Phase 1 recommendations selected per held-out pair.
PHASE1_RECOMMENDATION_BUDGET: int = 20

# Top-k metric requests. Regional per-region quotas are derived from the budget.
TRAIN_TOP_K_VALUES: list[int] = [PHASE1_RECOMMENDATION_BUDGET]

# NPD hit-rate thresholds reported in Phase 1.
TRAIN_NPD_THRESHOLDS: list[float] = [0.05, 0.10]

# Apply the held-out completed-configuration threshold before Phase 1 evaluation.
PHASE1_MIN_COMPLETED_ENABLED: bool = True

# Minimum completed held-out configurations required when the threshold is active.
PHASE1_MIN_COMPLETED_CONFIGS: int = 100


# =============================================================================
# Phase 1 evaluation: plotting
# =============================================================================
# Controls Pareto-space plots generated by Phase 1 evaluation.

# Generate linear-axis Pareto plots.
TRAIN_PLOT_PARETO: bool = True

# Generate log-runtime Pareto plots.
TRAIN_PLOT_PARETO_LOG: bool = True

# Matplotlib figure size for Phase 1 plots.
PLOT_FIGSIZE: tuple[int, int] = (10, 6)

# Image resolution for saved Phase 1 plots.
PLOT_SAVE_DPI: int = 200


# =============================================================================
# Configuration recommendation: shared settings
# =============================================================================
# Controls Phase 2 candidate generation from trained models and manual
# detector search-space CSVs.

# Recommendation refinement mode: static | dynamic.
CONFIG_RECOMMENDATION_MODE: str = "static"

# Exact final recommendation budget for each detector-dataset pair.
PHASE2_RECOMMENDATION_BUDGET: int = 20

# Recommendation target mode: single | separate.
CONFIG_RECOMMENDATION_TARGET_MODE: str = TRAIN_TARGET_MODE

# Detector/dataset scope for configuration recommendation.
CONFIG_RECOMMENDATION_DETECTORS: list[str] = TRAIN_DETECTORS.copy()
CONFIG_RECOMMENDATION_DATASETS: list[str] = TRAIN_DATASETS.copy()

# Random seed used by Sobol and Optuna samplers.
CONFIG_RECOMMENDATION_RANDOM_SEED: int = GLOBAL_RANDOM_SEED

# Outer recommendation parallelism.
CONFIG_RECOMMENDATION_N_JOBS: int = 1

# Results subdirectory under results/phase_1/.
CONFIG_RECOMMENDATION_RESULTS_DIR_NAME: str = "system_recommendations"

# Constraint-resampling attempts for invalid Sobol candidate draws.
CONFIG_RECOMMENDATION_SOBOL_MAX_CONSTRAINT_ATTEMPTS: int = 50

# Number of Sobol configurations generated before seed selection.
CONFIG_RECOMMENDATION_GLOBAL_SOBOL_SAMPLE_COUNT: int = 32_768

# Remove exact duplicate hyperparameter rows after Sobol sampling.
CONFIG_RECOMMENDATION_DROP_EXACT_DUPLICATE_CONFIGS_AFTER_SOBOL: bool = True

# Remove exact duplicate hyperparameter rows after Optuna refinement.
CONFIG_RECOMMENDATION_DROP_EXACT_DUPLICATE_CONFIGS_AFTER_OPTUNA: bool = True

# Number of random startup trials for TPE studies.
CONFIG_RECOMMENDATION_OPTUNA_STARTUP_TRIALS: int = 0

# Optuna per-study parallelism.
CONFIG_RECOMMENDATION_OPTUNA_STUDY_N_JOBS: int = 1

# Enable multivariate TPE sampling.
CONFIG_RECOMMENDATION_TPE_MULTIVARIATE: bool = True

# Enable grouped TPE sampling.
CONFIG_RECOMMENDATION_TPE_GROUP: bool = False

# Enable TPE constant-liar behavior for parallel suggestions.
CONFIG_RECOMMENDATION_TPE_CONSTANT_LIAR: bool = False

# Restrict Optuna numeric domains to semi-local bounds around selected seeds.
CONFIG_RECOMMENDATION_USE_SEMI_LOCAL_OPTUNA_SPACE: bool = True

# Lower quantile used when constructing semi-local numeric bounds.
CONFIG_RECOMMENDATION_LOCAL_NUMERIC_QUANTILE_LOW: float = 0.10

# Upper quantile used when constructing semi-local numeric bounds.
CONFIG_RECOMMENDATION_LOCAL_NUMERIC_QUANTILE_HIGH: float = 0.90

# Generate prediction-uncertainty diagnostics; does not affect selection.
COMPUTE_PREDICTION_UNCERTAINTY: bool = False

# Use a fresh TPE study for each dynamic margin stage.
CONFIG_RECOMMENDATION_USE_FRESH_STUDY_PER_MARGIN_STAGE: bool = True

# Restrict categorical domains to values present in selected seeds.
CONFIG_RECOMMENDATION_RESTRICT_CATEGORICALS_TO_SELECTED_SEEDS: bool = False

# Restrict Boolean domains to values present in selected seeds.
CONFIG_RECOMMENDATION_RESTRICT_BOOLEANS_TO_SELECTED_SEEDS: bool = False

# Add categorical/Boolean variants after Optuna when enabled.
CONFIG_RECOMMENDATION_EXPAND_CATEGORICAL_BOOL_COMBINATIONS_AFTER_OPTUNA: bool = False


# =============================================================================
# Configuration recommendation: static TPE refinement
# =============================================================================
# Static refinement uses one TPE stage. Warm-start seed evaluations are
# additional Optuna trials beyond the configured refinement-trial count.

# Number of Sobol seeds used as TPE warm starts.
STATIC_TPE_SEED_COUNT: int = 200

# Numeric margin ratio for static semi-local TPE bounds.
STATIC_TPE_NUMERIC_MARGIN: float = 0.10

# Number of TPE refinement trials after warm-start seeds.
STATIC_TPE_TRIALS: int = 500


# =============================================================================
# Configuration recommendation: dynamic TPE refinement
# =============================================================================
# Dynamic refinement uses three TPE stages with stage-wise seed reselection.
# Warm-start seed evaluations are additional Optuna trials.

# Numeric margin ratios for the three dynamic TPE stages.
DYNAMIC_TPE_MARGIN_SCHEDULE: tuple[float, float, float] = (0.15, 0.10, 0.05)

# Warm-start seed counts for the three dynamic TPE stages.
DYNAMIC_TPE_SEED_COUNT_SCHEDULE: tuple[int, int, int] = (300, 200, 100)

# Trial-budget shares assigned to the three dynamic TPE stages.
DYNAMIC_TPE_TRIAL_SHARE_SCHEDULE: tuple[float, float, float] = (0.20, 0.50, 0.30)

# Total TPE refinement trials distributed across dynamic stages.
DYNAMIC_TPE_TOTAL_TRIALS: int = 1_000


# =============================================================================
# Configuration recommendation: shared NSGA-II refinement for Separate
# =============================================================================
# Separate recommendation also runs an independent one-stage NSGA-II branch.

# Number of Sobol seeds used as NSGA-II warm starts.
NSGAII_SEED_COUNT: int = 200

# Numeric margin ratio for NSGA-II semi-local bounds.
NSGAII_NUMERIC_MARGIN: float = 0.10

# Number of NSGA-II refinement trials after warm-start seeds.
NSGAII_TRIALS: int = 1_000

# NSGA-II population size passed to Optuna.
NSGAII_POPULATION_SIZE: int = 50


# =============================================================================
# Pipeline runner setup
# =============================================================================
# Controls sequential execution by pipeline_runner.py. Stage dependencies are
# not inferred; enable only the stages that should be executed.

# Runner mode: single | sweep.
PIPELINE_RUN_MODE: str = "single"

# Run preprocessing_data.py.
PIPELINE_RUN_PREPROCESSING: bool = False

# Run metadata_extraction.py.
PIPELINE_RUN_METADATA_EXTRACTION: bool = False

# Run train_models.py.
PIPELINE_RUN_TRAINING: bool = True

# Run phase1_evaluation.py.
PIPELINE_RUN_PHASE1_EVALUATION: bool = True

# Run config_recommendation.py.
PIPELINE_RUN_CONFIG_RECOMMENDATION: bool = True

# Target formulations expanded in runner sweep mode.
PIPELINE_SWEEP_TARGET_FORMULATIONS: list[str] = [
    "mod_dist",
    "tchebycheff",
    "pbi",
    "apd",
    "separate",
]

# Metadata setups expanded in runner sweep mode.
PIPELINE_SWEEP_METADATA_SETUPS: list[tuple[str, float | None]] = [
    ("cfg", None),
    ("pruned", None),
    ("lodo_pca", 0.80),
    ("lodo_pca", 0.85),
    ("lodo_pca", 0.90),
    ("lodo_pca", 0.95),
]

