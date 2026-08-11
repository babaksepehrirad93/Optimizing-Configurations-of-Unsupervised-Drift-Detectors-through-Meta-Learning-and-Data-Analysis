"""Pipeline execution controller.

Executes the enabled preprocessing, metadata extraction, training, Phase 1
evaluation, and configuration recommendation stages in pipeline order. Stage
selection and runner mode are configured in `src.config`.

In `single` mode, setup settings are read directly from `src.config`. In
`sweep` mode, configured target x metadata combinations are passed to
setup-aware child processes. Detector/dataset scope is provided through CLI.

Usage:
    python pipeline_runner.py --detector ALL --dataset ALL
    python pipeline_runner.py --dry-run --detector BNDM --dataset Electricity
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    ALL_DATASETS,
    ALL_DETECTORS,
    PCA_LODO_VARIANCE,
    PCA_VARIANCES,
    PIPELINE_RUN_CONFIG_RECOMMENDATION,
    PIPELINE_RUN_METADATA_EXTRACTION,
    PIPELINE_RUN_MODE,
    PIPELINE_RUN_PHASE1_EVALUATION,
    PIPELINE_RUN_PREPROCESSING,
    PIPELINE_RUN_TRAINING,
    PIPELINE_SWEEP_METADATA_SETUPS,
    PIPELINE_SWEEP_TARGET_FORMULATIONS,
    SINGLE_TARGET_FORMULATION,
    SUPPORTED_SINGLE_TARGET_FORMULATIONS,
    TRAIN_METADATA_VARIANT,
    TRAIN_TARGET_MODE,
    TRAIN_USE_METADATA,
)
from src.paths import get_paths_from_script
from src.sweeper_setup import PCA_METADATA_VARIANTS, SUPPORTED_METADATA_VARIANTS


@dataclass(frozen=True)
class PipelineSetup:
    target: str
    target_mode: str
    single_target_formulation: str | None
    metadata_variant: str
    pca_variance: float | None

    @property
    def metadata_label(self) -> str:
        if self.metadata_variant == "cfg":
            return "CFG"
        if self.metadata_variant in PCA_METADATA_VARIANTS:
            return f"{self.metadata_variant} {float(self.pca_variance):.2f}"
        return self.metadata_variant


@dataclass(frozen=True)
class PipelineStage:
    label: str
    script_name: str
    enabled: bool
    accepts_scope: bool
    accepts_setup: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run enabled thesis pipeline stages.")
    parser.add_argument("--detector", type=str, default="ALL", help="Detector name, ALL, or comma-separated exact names.")
    parser.add_argument("--dataset", type=str, default="ALL", help="Dataset name, ALL, or comma-separated exact names.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned setup/stage commands without executing them.")
    return parser.parse_args()


def _resolve_cli_selection(selected: str, *, available: list[str], what: str) -> tuple[list[str], str]:
    token = str(selected).strip()
    if not token:
        raise ValueError(f"Empty {what} selection is not allowed.")
    if token.upper() == "ALL":
        return list(available), "ALL"
    requested = [part.strip() for part in token.split(",") if part.strip()]
    if not requested:
        raise ValueError(f"Empty {what} selection is not allowed.")
    unknown = [item for item in requested if item not in available]
    if unknown:
        raise ValueError(f"Unknown {what}(s) {unknown}. Available: {available}")
    seen: set[str] = set()
    ordered = []
    for item in requested:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered, ",".join(ordered)


def _validate_pca_variance(value: float | None, *, metadata_variant: str) -> float | None:
    if metadata_variant not in PCA_METADATA_VARIANTS:
        if value is not None:
            raise ValueError(f"Metadata setup {metadata_variant!r} must not specify a PCA variance.")
        return None
    if value is None:
        raise ValueError(f"Metadata setup {metadata_variant!r} requires a PCA variance.")
    allowed = [float(candidate) for candidate in PCA_VARIANCES]
    if not any(abs(float(value) - candidate) <= 1e-12 for candidate in allowed):
        raise ValueError(f"PCA variance for {metadata_variant!r} must be one of {allowed}. Got {value!r}.")
    return float(value)


def _metadata_setup(variant: str, variance: float | None) -> tuple[str, float | None]:
    metadata_variant = str(variant).strip().lower()
    if metadata_variant not in SUPPORTED_METADATA_VARIANTS:
        raise ValueError(f"Unknown metadata setup {metadata_variant!r}. Supported: {sorted(SUPPORTED_METADATA_VARIANTS)}.")
    pca_variance = _validate_pca_variance(variance, metadata_variant=metadata_variant)
    return metadata_variant, pca_variance


def _setup_from_target_and_metadata(target: str, metadata_variant: str, pca_variance: float | None) -> PipelineSetup:
    target_name = str(target).strip()
    if target_name == "separate":
        return PipelineSetup(
            target="separate",
            target_mode="separate",
            single_target_formulation=None,
            metadata_variant=metadata_variant,
            pca_variance=pca_variance,
        )
    if target_name not in SUPPORTED_SINGLE_TARGET_FORMULATIONS:
        raise ValueError(
            f"Unknown sweep target formulation {target_name!r}. "
            f"Use 'separate' or one of {list(SUPPORTED_SINGLE_TARGET_FORMULATIONS)}."
        )
    return PipelineSetup(
        target=target_name,
        target_mode="single",
        single_target_formulation=target_name,
        metadata_variant=metadata_variant,
        pca_variance=pca_variance,
    )


def _configured_single_metadata() -> tuple[str, float | None]:
    target_mode = str(TRAIN_TARGET_MODE).strip().lower()
    if target_mode not in {"single", "separate"}:
        raise ValueError(f"PIPELINE single mode requires TRAIN_TARGET_MODE to be 'single' or 'separate'. Got {TRAIN_TARGET_MODE!r}.")
    metadata_variant = "cfg" if not TRAIN_USE_METADATA else str(TRAIN_METADATA_VARIANT).strip().lower()
    pca_variance = float(PCA_LODO_VARIANCE) if metadata_variant in PCA_METADATA_VARIANTS else None
    return _metadata_setup(metadata_variant, pca_variance)


def _sweep_setups() -> list[PipelineSetup]:
    metadata_setups = [_metadata_setup(name, variance) for name, variance in PIPELINE_SWEEP_METADATA_SETUPS]
    setups = [
        _setup_from_target_and_metadata(target, metadata_variant, pca_variance)
        for target in PIPELINE_SWEEP_TARGET_FORMULATIONS
        for metadata_variant, pca_variance in metadata_setups
    ]
    if not setups:
        raise ValueError("PIPELINE_SWEEP_TARGET_FORMULATIONS x PIPELINE_SWEEP_METADATA_SETUPS produced no setups.")
    return setups


def resolve_setups() -> tuple[str, list[PipelineSetup | None]]:
    mode = str(PIPELINE_RUN_MODE).strip().lower()
    if mode == "single":
        _configured_single_metadata()
        return mode, [None]
    if mode == "sweep":
        return mode, _sweep_setups()
    raise ValueError(f"PIPELINE_RUN_MODE must be 'single' or 'sweep'. Got {PIPELINE_RUN_MODE!r}.")


def enabled_stages() -> list[PipelineStage]:
    stages = [
        PipelineStage("preprocessing", "preprocessing_data.py", PIPELINE_RUN_PREPROCESSING, True, False),
        PipelineStage("metadata extraction", "metadata_extraction.py", PIPELINE_RUN_METADATA_EXTRACTION, False, False),
        PipelineStage("training", "train_models.py", PIPELINE_RUN_TRAINING, True, True),
        PipelineStage("Phase 1 evaluation", "phase1_evaluation.py", PIPELINE_RUN_PHASE1_EVALUATION, True, True),
        PipelineStage("configuration recommendation", "config_recommendation.py", PIPELINE_RUN_CONFIG_RECOMMENDATION, True, True),
    ]
    active = [stage for stage in stages if bool(stage.enabled)]
    if not active:
        raise ValueError("At least one PIPELINE_RUN_* stage toggle must be enabled.")
    return active


def _setup_args(setup: PipelineSetup) -> list[str]:
    args = ["--pipeline-target-mode", setup.target_mode]
    if setup.single_target_formulation is not None:
        args.extend(["--pipeline-target-formulation", setup.single_target_formulation])
    args.extend(["--pipeline-metadata-variant", setup.metadata_variant])
    if setup.pca_variance is not None:
        args.extend(["--pipeline-pca-variance", f"{setup.pca_variance:.2f}"])
    return args


def build_command(
    *,
    project_root: Path,
    stage: PipelineStage,
    setup: PipelineSetup | None,
    run_mode: str,
    detector_arg: str,
    dataset_arg: str,
) -> list[str]:
    command = [sys.executable, str(project_root / stage.script_name)]
    if stage.accepts_scope:
        command.extend(["--detector", detector_arg, "--dataset", dataset_arg])
    if run_mode == "sweep" and stage.accepts_setup:
        if setup is None:
            raise ValueError("Sweep mode requires a generated setup for setup-aware stages.")
        command.extend(_setup_args(setup))
    return command


def _format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


def _format_duration(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _print_runner_summary(
    *,
    mode: str,
    detectors: list[str],
    datasets: list[str],
    stages: list[PipelineStage],
    setup_count: int,
) -> None:
    print("Pipeline runner")
    print(f"Mode: {mode}")
    print("Detectors: " + ", ".join(detectors))
    print("Datasets: " + ", ".join(datasets))
    if mode == "single":
        metadata_variant, pca_variance = _configured_single_metadata()
        print(f"Configured target mode: {TRAIN_TARGET_MODE}")
        if str(TRAIN_TARGET_MODE).strip().lower() == "single":
            print(f"Configured target formulation: {SINGLE_TARGET_FORMULATION}")
        else:
            print("Configured target formulation: separate")
        print(f"Configured metadata: {metadata_variant}")
        print(f"Configured PCA variance: {pca_variance if pca_variance is not None else 'not_applicable'}")
    print("Enabled stages:")
    active_names = {stage.label for stage in stages}
    for stage in [
        "preprocessing",
        "metadata extraction",
        "training",
        "Phase 1 evaluation",
        "configuration recommendation",
    ]:
        print(f"  {stage}: {'yes' if stage in active_names else 'no'}")
    print(f"Setups: {setup_count}")


def _print_setup_header(setup: PipelineSetup | None, *, index: int, total: int, mode: str) -> None:
    print("\n" + "=" * 60)
    print(f"{'Setup' if mode == 'sweep' else 'Run'} {index}/{total}")
    if setup is None:
        metadata_variant, pca_variance = _configured_single_metadata()
        print(f"Target mode: {TRAIN_TARGET_MODE}")
        if str(TRAIN_TARGET_MODE).strip().lower() == "single":
            print(f"Target formulation: {SINGLE_TARGET_FORMULATION}")
        else:
            print("Target formulation: separate")
        print(f"Metadata: {metadata_variant}")
        print(f"PCA variance: {pca_variance if pca_variance is not None else 'not_applicable'}")
        print("=" * 60)
        return
    print(f"Target mode: {setup.target_mode}")
    if setup.target_mode == "single":
        print(f"Target formulation: {setup.single_target_formulation}")
    else:
        print("Target formulation: separate")
    print(f"Metadata: {setup.metadata_variant}")
    print(f"PCA variance: {setup.pca_variance if setup.pca_variance is not None else 'not_applicable'}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    detectors, detector_arg = _resolve_cli_selection(args.detector, available=ALL_DETECTORS, what="detector")
    datasets, dataset_arg = _resolve_cli_selection(args.dataset, available=ALL_DATASETS, what="dataset")
    mode, setups = resolve_setups()
    stages = enabled_stages()
    paths = get_paths_from_script(__file__)
    project_root = paths.project_root

    _print_runner_summary(
        mode=mode,
        detectors=detectors,
        datasets=datasets,
        stages=stages,
        setup_count=len(setups),
    )

    started_at = time.perf_counter()
    executed_stage_count = 0
    for setup_index, setup in enumerate(setups, start=1):
        _print_setup_header(setup, index=setup_index, total=len(setups), mode=mode)
        for stage_index, stage in enumerate(stages, start=1):
            command = build_command(
                project_root=project_root,
                stage=stage,
                setup=setup,
                run_mode=mode,
                detector_arg=detector_arg,
                dataset_arg=dataset_arg,
            )
            print(f"[{stage_index}/{len(stages)}] {stage.script_name}")
            print("  " + _format_command(command))
            if args.dry_run:
                continue
            try:
                subprocess.run(command, cwd=project_root, check=True)
            except subprocess.CalledProcessError as exc:
                print("\nPipeline stage failed.")
                if setup is None:
                    print(f"Run: {setup_index}/{len(setups)} using src.config")
                else:
                    print(f"Setup: {setup_index}/{len(setups)} target={setup.target} metadata={setup.metadata_label}")
                print(f"Stage: {stage.label} ({stage.script_name})")
                print("Command: " + _format_command(command))
                print(f"Return code: {exc.returncode}")
                raise SystemExit(exc.returncode) from exc
            executed_stage_count += 1

    if args.dry_run:
        print(f"\nDry run complete. Planned setups: {len(setups)}")
        print(f"Planned stage executions: {len(setups) * len(stages)}")
        return

    print("\nPipeline runner complete.")
    print(f"Completed setups: {len(setups)}")
    print(f"Total stages executed: {executed_stage_count}")
    print(f"Total runtime: {_format_duration(time.perf_counter() - started_at)}")


if __name__ == "__main__":
    main()
