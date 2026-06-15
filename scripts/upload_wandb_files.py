#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload one file or a directory of files to W&B as an artifact.")
    parser.add_argument("--project", default="maze-hrm", help="W&B project name.")
    parser.add_argument("--entity", help="Optional W&B entity/team.")
    parser.add_argument("--artifact-name", required=True, help="Artifact name to create.")
    parser.add_argument("--artifact-type", default="analysis", help="Artifact type, e.g. analysis, model, dataset.")
    parser.add_argument("--path", required=True, help="File or directory to upload.")
    parser.add_argument("--run-name", help="Readable W&B run name for this upload.")
    parser.add_argument("--recursive", action="store_true", help="Upload files recursively when --path is a directory.")
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Glob relative to --path to exclude. May be passed more than once.",
    )
    return parser.parse_args()


def iter_files(path: Path, recursive: bool, exclude_globs: list[str] | None = None) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")
    pattern = "**/*" if recursive else "*"
    exclude_globs = exclude_globs or []
    return sorted(
        candidate
        for candidate in path.glob(pattern)
        if candidate.is_file()
        and not any(candidate.relative_to(path).match(exclude) for exclude in exclude_globs)
    )


def main() -> None:
    args = parse_args()
    source = Path(args.path).resolve()
    files = iter_files(source, args.recursive, args.exclude_glob)
    if not files:
        raise ValueError(f"No files found to upload from {source}")

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name or f"artifact-{args.artifact_name}",
        job_type="artifact-upload",
    )
    artifact = wandb.Artifact(args.artifact_name, type=args.artifact_type)
    for file_path in files:
        if source.is_dir():
            artifact.add_file(str(file_path), name=str(file_path.relative_to(source)))
        else:
            artifact.add_file(str(file_path))

    run.log_artifact(artifact)
    run.finish()

    print(f"Uploaded artifact '{args.artifact_name}' with {len(files)} file(s).")


if __name__ == "__main__":
    main()
