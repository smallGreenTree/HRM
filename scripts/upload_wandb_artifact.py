#!/usr/bin/env python3
import argparse
from pathlib import Path

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a checkpoint directory to Weights & Biases as a model artifact.")
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument("--artifact-name", required=True, help="Artifact name to create in W&B.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing step_* checkpoints and metadata files.")
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Specific checkpoint filename or absolute path. Defaults to the latest step_* file in checkpoint-dir.",
    )
    parser.add_argument(
        "--metadata-file",
        action="append",
        default=[],
        help="Additional file to attach to the artifact. May be passed multiple times.",
    )
    return parser.parse_args()


def resolve_checkpoint_file(checkpoint_dir: Path, checkpoint_file: str | None) -> Path:
    if checkpoint_file is not None:
        candidate = Path(checkpoint_file)
        return candidate if candidate.is_absolute() else checkpoint_dir / candidate

    checkpoint_files = sorted(
        path
        for path in checkpoint_dir.iterdir()
        if path.name.startswith("step_") and "_all_preds" not in path.name
    )
    if not checkpoint_files:
        raise FileNotFoundError(f"No step_* checkpoint files found in {checkpoint_dir}")
    return checkpoint_files[-1]


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    checkpoint_file = resolve_checkpoint_file(checkpoint_dir, args.checkpoint_file).resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_file}")

    files_to_add = [checkpoint_file]
    default_metadata = ["all_config.yaml", "eval_metrics.csv", "train_metrics.csv"]
    for name in default_metadata:
        candidate = checkpoint_dir / name
        if candidate.exists():
            files_to_add.append(candidate.resolve())

    for name in args.metadata_file:
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = checkpoint_dir / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {candidate}")
        files_to_add.append(candidate)

    run = wandb.init(project=args.project, job_type="artifact-upload")
    artifact = wandb.Artifact(args.artifact_name, type="model")
    for file_path in files_to_add:
        artifact.add_file(str(file_path))
    run.log_artifact(artifact)
    run.finish()

    print(f"Uploaded artifact '{args.artifact_name}' with checkpoint '{checkpoint_file.name}'.")


if __name__ == "__main__":
    main()
