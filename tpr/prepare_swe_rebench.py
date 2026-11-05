#!/usr/bin/env python3
"""Prepare SWE-ReBench task data in Uni-Agent's native Task Config format.

By default this mirrors Uni-Agent's upstream preprocessing and writes the full
filtered SWE-ReBench split. Use --max-instances for a small prefix, or
--instance-ids-file to select an explicit local subset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uni_agent.tasks.swe_rebench.preprocess import build_swe_rebench, get_image_name


def _read_instance_ids(path: Path) -> list[str]:
    instance_ids = [
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    instance_ids = [instance_id for instance_id in instance_ids if instance_id]
    if not instance_ids:
        raise ValueError(f"no instance IDs found in {path}")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError(f"duplicate instance IDs found in {path}")
    return instance_ids


def _task_metadata(row: dict) -> dict:
    return row["extra_info"]["tools_kwargs"]["task"]["metadata"]


def _select_instance_ids(dataset, instance_ids: list[str]):
    """Select requested rows in the same order as the ID list."""
    positions: dict[str, int] = {}
    requested = set(instance_ids)
    for index, row in enumerate(dataset):
        instance_id = _task_metadata(row)["instance_id"]
        if instance_id in requested:
            if instance_id in positions:
                raise ValueError(f"dataset contains duplicate instance_id={instance_id!r}")
            positions[instance_id] = index

    missing = [instance_id for instance_id in instance_ids if instance_id not in positions]
    if missing:
        raise ValueError(f"instance IDs not found in nebius/SWE-rebench filtered split: {missing}")

    return dataset.select([positions[instance_id] for instance_id in instance_ids])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--instance-id", help="Select one instance (backward-compatible shortcut).")
    selector.add_argument(
        "--instance-ids-file",
        type=Path,
        help="Text file with one instance_id per line; blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Keep only the first N rows of the filtered split (upstream Uni-Agent behavior).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.max_instances is not None and args.max_instances < 0:
        parser.error("--max-instances must be non-negative")
    if args.max_instances is not None and (args.instance_id or args.instance_ids_file):
        parser.error("--max-instances cannot be combined with an explicit instance selector")

    # Keep dataset loading, row conversion, and image naming aligned with the
    # pinned Uni-Agent implementation instead of maintaining a second converter.
    dataset = build_swe_rebench(max_instances=args.max_instances)

    instance_ids = None
    if args.instance_id:
        instance_ids = [args.instance_id]
    elif args.instance_ids_file:
        instance_ids = _read_instance_ids(args.instance_ids_file)

    if instance_ids is not None:
        dataset = _select_instance_ids(dataset, instance_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(args.output))

    print(f"Wrote {len(dataset)} instances to {args.output.resolve()}", flush=True)
    for row in dataset:
        metadata = _task_metadata(row)
        instance_id = metadata["instance_id"]
        print(f"  {instance_id}: {get_image_name(instance_id)}", flush=True)


if __name__ == "__main__":
    main()
