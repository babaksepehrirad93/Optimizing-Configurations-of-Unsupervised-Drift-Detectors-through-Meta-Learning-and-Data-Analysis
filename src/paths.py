"""Centralized project path construction.

Derives all data, model, Phase 1 result, Phase 2 result, metadata, and
search-space paths from the repository root. Scripts use these helpers so
artifact locations do not depend on the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProjectPaths:
    """
    Central path container for the thesis project.

    Only project_root is stored directly.
    All other directories are derived dynamically from it so the path logic
    stays consistent across all scripts.
    """

    project_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())

    # ------------------------------------------------------------------
    # Core directories
    # ------------------------------------------------------------------

    @property
    def src_dir(self) -> Path:
        return self.project_root / "src"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw_data"

    @property
    def raw_benchmarking_results_dir(self) -> Path:
        return self.raw_data_dir / "benchmarking_results"

    @property
    def raw_datasets_dir(self) -> Path:
        return self.raw_data_dir / "datasets"

    @property
    def raw_detector_default_configs_dir(self) -> Path:
        return self.raw_data_dir / "detector_default_configs"

    @property
    def extracted_metadata_dir(self) -> Path:
        return self.data_dir / "extracted_metadata"

    @property
    def processed_benchmark_data_dir(self) -> Path:
        return self.data_dir / "processed_benchmark_data"

    @property
    def search_space_dir(self) -> Path:
        return self.raw_data_dir / "search_space"

    @property
    def training_data_dir(self) -> Path:
        return self.data_dir / "training_data"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def results_phase1_dir(self) -> Path:
        return self.results_dir / "phase_1"

    @property
    def held_out_evaluation_dir(self) -> Path:
        return self.results_phase1_dir / "held-out_evaluation"

    @property
    def system_recommendations_dir(self) -> Path:
        return self.results_phase1_dir / "system_recommendations"

    # ------------------------------------------------------------------
    # Metadata files / folders
    # ------------------------------------------------------------------

    @property
    def dataset_overview_file(self) -> Path:
        return self.extracted_metadata_dir / "dataset_overview.csv"

    @property
    def all_meta_features_file(self) -> Path:
        return self.extracted_metadata_dir / "all_meta_features.csv"

    @property
    def cleaned_meta_features_file(self) -> Path:
        return self.extracted_metadata_dir / "cleaned_meta_features.csv"

    @property
    def pruned_meta_features_file(self) -> Path:
        return self.extracted_metadata_dir / "pruned_meta_features.csv"

    @property
    def metadata_pca_dir(self) -> Path:
        return self.extracted_metadata_dir / "pca"

    @property
    def metadata_pca_lodo_dir(self) -> Path:
        return self.metadata_pca_dir / "lodo_pca"

    @property
    def metadata_pca_ranked_dir(self) -> Path:
        return self.metadata_pca_dir / "ranked"

    @property
    def metadata_pca_variance_info_dir(self) -> Path:
        return self.metadata_pca_dir / "variance_info"

    @staticmethod
    def pca_variance_tag(variance: float) -> str:
        return f"{float(variance):.2f}"

    def metadata_pca_lodo_variant_dir(self, variance: float) -> Path:
        return self.metadata_pca_lodo_dir / f"lodo_pca_{self.pca_variance_tag(variance)}"

    def metadata_pca_lodo_file(self, dataset_name: str, variance: float) -> Path:
        return self.metadata_pca_lodo_variant_dir(variance) / f"pca_meta_features_lodo_{dataset_name}.csv"

    def metadata_pca_variance_info_lodo_file(self, dataset_name: str, variance: float) -> Path:
        return (
            self.metadata_pca_variance_info_dir
            / f"lodo_pca_{self.pca_variance_tag(variance)}"
            / f"pca_variance_info_lodo_{dataset_name}.csv"
        )

    def metadata_pca_ranked_lodo_variant_dir(self, variance: float) -> Path:
        return self.metadata_pca_ranked_dir / f"lodo_pca_{self.pca_variance_tag(variance)}"

    def metadata_pca_ranked_lodo_file(self, dataset_name: str, variance: float) -> Path:
        return self.metadata_pca_ranked_lodo_variant_dir(variance) / f"pca_ranked_meta_features_lodo_{dataset_name}.csv"

    # ------------------------------------------------------------------
    # Raw benchmark data helpers
    # ------------------------------------------------------------------

    def raw_detector_dir(self, detector_name: str) -> Path:
        return self.raw_benchmarking_results_dir / detector_name

    def raw_benchmark_file(self, detector_name: str, dataset_name: str) -> Path:
        return self.raw_detector_dir(detector_name) / f"{detector_name}_{dataset_name}.csv"

    def raw_detector_default_configs_file(self, detector_name: str) -> Path:
        return self.raw_detector_default_configs_dir / f"{detector_name}_default_configs.csv"

    def raw_detector_dataset_default_configs_file(self, detector_name: str, dataset_name: str) -> Path:
        return (
            self.raw_detector_default_configs_dir
            / detector_name
            / f"{detector_name}_{dataset_name}_default_configs.csv"
        )

    # ------------------------------------------------------------------
    # Search-space helpers
    # ------------------------------------------------------------------

    def detector_search_space_file(self, detector_name: str) -> Path:
        return self.search_space_dir / f"{detector_name}_search_space.csv"

    # ------------------------------------------------------------------
    # Processed benchmark data helpers
    # ------------------------------------------------------------------

    def processed_detector_dir(self, detector_name: str) -> Path:
        return self.processed_benchmark_data_dir / detector_name

    def processed_benchmark_file(self, detector_name: str, dataset_name: str) -> Path:
        return self.processed_detector_dir(detector_name) / f"{detector_name}_{dataset_name}_processed.csv"

    # ------------------------------------------------------------------
    # Training-data helpers
    # ------------------------------------------------------------------

    def training_snapshot_file(self, filename: str) -> Path:
        return self.training_data_dir / filename

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def detector_models_dir(self, detector_name: str) -> Path:
        return self.models_dir / detector_name

    def lodo_model_file(self, detector_name: str, dataset_name: str, *, suffix: str = "") -> Path:
        suffix_part = f"_{suffix}" if suffix else ""
        return self.detector_models_dir(detector_name) / f"{detector_name}_LODO_{dataset_name}{suffix_part}.joblib"

    @staticmethod
    def phase1_artifact_tag(
        *,
        target_mode: str,
        target_method: str,
        metadata_tag: str,
        preference_region: str | None = None,
        objective: str | None = None,
    ) -> str:
        """Build the stable tag shared by Phase 1 training and evaluation artifacts."""
        parts = [
            str(target_mode).strip().lower(),
            str(target_method).strip(),
            str(metadata_tag).strip(),
        ]
        if preference_region:
            parts.append(str(preference_region).strip())
        if objective:
            parts.append(str(objective).strip())
        safe_parts: list[str] = []
        for part in parts:
            safe = part.replace(" ", "_")
            safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in safe)
            while "__" in safe:
                safe = safe.replace("__", "_")
            safe_parts.append(safe.strip("_") or "untitled")
        return "__".join(safe_parts)

    def phase1_model_file(
        self,
        detector_name: str,
        dataset_name: str,
        *,
        target_mode: str,
        target_method: str,
        metadata_tag: str,
        preference_region: str | None = None,
        objective: str | None = None,
    ) -> Path:
        """Return the unique saved-model path for one Phase 1 LODO artifact."""
        artifact_tag = self.phase1_artifact_tag(
            target_mode=target_mode,
            target_method=target_method,
            metadata_tag=metadata_tag,
            preference_region=preference_region,
            objective=objective,
        )
        return (
            self.detector_models_dir(detector_name)
            / "phase1"
            / artifact_tag
            / f"{detector_name}_LODO_{dataset_name}_{artifact_tag}.joblib"
        )

    def phase1_model_metadata_file(
        self,
        detector_name: str,
        dataset_name: str,
        *,
        target_mode: str,
        target_method: str,
        metadata_tag: str,
        preference_region: str | None = None,
        objective: str | None = None,
    ) -> Path:
        """Return the sidecar metadata path for a Phase 1 saved model."""
        return self.phase1_model_file(
            detector_name,
            dataset_name,
            target_mode=target_mode,
            target_method=target_method,
            metadata_tag=metadata_tag,
            preference_region=preference_region,
            objective=objective,
        ).with_suffix(".json")

    def phase1_training_snapshot_file(
        self,
        detector_name: str,
        dataset_name: str,
        *,
        target_mode: str,
        target_method: str,
        metadata_tag: str,
        preference_region: str | None = None,
    ) -> Path:
        """Return the training-data snapshot path for one Phase 1 LODO setup."""
        artifact_tag = self.phase1_artifact_tag(
            target_mode=target_mode,
            target_method=target_method,
            metadata_tag=metadata_tag,
            preference_region=preference_region,
        )
        return (
            self.training_data_dir
            / detector_name
            / "phase1"
            / artifact_tag
            / f"{detector_name}_LODO_{dataset_name}_{artifact_tag}_training_snapshot.csv"
        )

    def phase1_result_dir(self, experiment_parent: str, experiment_tag: str) -> Path:
        """Return the root output directory for one Phase 1 evaluation setup."""
        return self.held_out_evaluation_subdir(experiment_parent, experiment_tag)

    def phase1_details_file(self, experiment_parent: str, experiment_tag: str, detector_name: str, dataset_name: str) -> Path:
        """Return the held-out details CSV path for one Phase 1 evaluation pair."""
        return (
            self.phase1_result_dir(experiment_parent, experiment_tag)
            / "details"
            / detector_name
            / f"{detector_name}_LODO_{dataset_name}_testing_details.csv"
        )

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    def held_out_evaluation_subdir(self, *parts: str) -> Path:
        path = self.held_out_evaluation_dir
        for part in parts:
            path = path / part
        return path

    def held_out_evaluation_file(self, *parts: str, filename: str) -> Path:
        return self.held_out_evaluation_subdir(*parts) / filename

    def system_recommendation_file(
        self,
        detector_name: str,
        dataset_name: str,
        *,
        format_tag: str,
        results_dir_name: str = "system_recommendations",
    ) -> Path:
        base_dir = self.results_phase1_dir / results_dir_name / detector_name / dataset_name
        if format_tag == "single":
            return base_dir / "single" / f"{detector_name}_{dataset_name}_single_recommended_configs.csv"
        if format_tag == "preference_regions":
            return (
                base_dir
                / "preference_regions_combined"
                / f"{detector_name}_{dataset_name}_final_recommended_configs.csv"
            )
        if format_tag == "separate":
            return (
                base_dir
                / "separate"
                / f"{detector_name}_{dataset_name}_separate_recommended_configs.csv"
            )
        raise ValueError("format_tag must be 'single', 'preference_regions' or 'separate'.")

    # ------------------------------------------------------------------
    # Directory creation
    # ------------------------------------------------------------------

    def ensure_core_directories(self) -> None:
        """
        Create the main project directories if they do not exist yet.
        Safe to call repeatedly.
        """
        directories = [
            self.src_dir,
            self.data_dir,
            self.raw_data_dir,
            self.raw_benchmarking_results_dir,
            self.raw_datasets_dir,
            self.raw_detector_default_configs_dir,
            self.extracted_metadata_dir,
            self.metadata_pca_dir,
            self.metadata_pca_lodo_dir,
            self.metadata_pca_ranked_dir,
            self.metadata_pca_variance_info_dir,
            self.processed_benchmark_data_dir,
            self.search_space_dir,
            self.training_data_dir,
            self.models_dir,
            self.results_phase1_dir,
            self.results_dir,
            self.held_out_evaluation_dir,
            self.system_recommendations_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def get_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Resolve the project root by walking upward until a directory containing
    both 'src' and 'data' is found.

    If nothing matches, return the resolved start directory.
    """
    start = Path(start_path).resolve() if start_path is not None else Path.cwd().resolve()

    if start.is_file():
        start = start.parent

    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate

    return start


def get_paths(project_root: Optional[Path] = None) -> ProjectPaths:
    """
    Build a ProjectPaths object from an optional project root.
    If not provided, the root is inferred from the current working directory.
    """
    root = get_project_root(project_root)
    return ProjectPaths(project_root=root)


def get_paths_from_script(script_file: str | Path) -> ProjectPaths:
    """
    Build a ProjectPaths object starting from a script file path such as __file__.
    """
    return get_paths(Path(script_file).resolve())
