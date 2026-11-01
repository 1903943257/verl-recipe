#!/usr/bin/env python3
"""Validate task sandbox image mapping and native Docker architectures before training."""

from __future__ import annotations

import argparse
import ast
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PELITA_IMAGE = "swerebench/sweb.eval.x86_64.aspp_1776_pelita-875"
PELITA_IMAGE_ID = "sha256:933103ac2f63e4d3ffcc8ffe769dfa38c7d4e85efc6ccfc3dcdba0b93617eaa6"


def normalize_arch(value: str) -> str:
    value = value.lower().strip()
    aliases = {
        "x86_64": "amd64",
        "x64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    return aliases.get(value, value)


def parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError(f"Expected a mapping in extra_info, got {type(value).__name__}")


def task_images(parquet_path: Path, task_config: Path) -> list[str]:
    """Resolve canonical sample images through Uni-Agent's image_map."""
    import pandas as pd
    from uni_agent.sandbox import SandboxConfig
    from uni_agent.tasks.config import TaskConfigResolver

    resolver = TaskConfigResolver.from_file(str(task_config))
    frame = pd.read_parquet(parquet_path)
    if "extra_info" not in frame.columns:
        raise ValueError("Parquet is missing extra_info")

    images: set[str] = set()
    for index, raw_info in enumerate(frame["extra_info"]):
        info = parse_mapping(raw_info)
        task = (info.get("tools_kwargs") or {}).get("task")
        if not isinstance(task, dict):
            raise ValueError(f"row {index}: missing task config")
        sample_sandbox = task.get("sandbox") or {}
        original = sample_sandbox.get("image")
        if not isinstance(original, str) or not original.startswith("swerebench/"):
            raise ValueError(f"row {index}: expected original SWE-ReBench image, got {original!r}")
        resolved = resolver.resolve(task)
        sandbox = SandboxConfig.model_validate(resolved["sandbox"])
        if sandbox.provider != "docker" or not sandbox.image:
            raise ValueError(f"row {index}: expected Docker sandbox with an image")
        print(f"[image-check] row {index}: {original} -> {sandbox.image}")
        images.add(sandbox.image)

    if not images:
        raise ValueError("No task images found in the Parquet")
    return sorted(images)


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def image_metadata(image: str) -> tuple[str, str, str]:
    result = run(
        ["docker", "image", "inspect", "--format", "{{.Id}}|{{.Os}}|{{.Architecture}}", image],
        capture=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Cannot inspect image {image}: {result.stderr.strip()}")
    image_id, os_name, arch = result.stdout.strip().split("|", maxsplit=2)
    return image_id, os_name, normalize_arch(arch)


def is_pinned_pelita_image(image: str) -> bool:
    return image in {PELITA_IMAGE, f"{PELITA_IMAGE}:latest"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--platform", default=None)
    args = parser.parse_args()

    if not args.parquet.is_file():
        parser.error(f"Parquet file not found: {args.parquet}")


    try:
        daemon = run(["docker", "info", "--format", "{{.OSType}}/{{.Architecture}}"], capture=True)
        if daemon.returncode != 0:
            raise RuntimeError(f"Docker daemon is unavailable: {daemon.stderr.strip()}")
        daemon_os, daemon_arch = daemon.stdout.strip().split("/", maxsplit=1)
        daemon_arch = normalize_arch(daemon_arch)
        host_arch = normalize_arch(platform.machine())
        platform_name = args.platform or f"{daemon_os}/{daemon_arch}"
        expected_os, expected_arch = platform_name.split("/", maxsplit=1)
        expected_arch = normalize_arch(expected_arch)

        print(f"[image-check] Docker host: {daemon_os}/{daemon_arch}")
        print(f"[image-check] Runtime host: {platform.system().lower()}/{host_arch}")
        print(f"[image-check] Requested task platform: {expected_os}/{expected_arch}")

        images = task_images(args.parquet, args.task_config)
        print(f"[image-check] Found {len(images)} unique task image(s) in {args.parquet}")

        for image in images:
            print(f"\n[image-check] Image: {image}")
            inspected = run(["docker", "image", "inspect", "--format", "{{.Id}}", image], capture=True)
            if inspected.returncode != 0:
                if not image.startswith("swerebench/"):
                    raise RuntimeError(f"Mapped image {image!r} is missing: load the verified ARM64 image first")
                if daemon_arch != "amd64":
                    raise RuntimeError(
                        f"Official SWE-ReBench is AMD64, but Docker host is {daemon_arch}; "
                        "select the ARM64 task config"
                    )
                pulled = run(["docker", "pull", "--platform", "linux/amd64", image])
                if pulled.returncode != 0:
                    raise RuntimeError(f"docker pull failed for {image}")

            image_id, image_os, image_arch = image_metadata(image)
            actual_platform = f"{image_os}/{image_arch}"
            print(f"[image-check] Image ID: {image_id}")
            print(f"[image-check] Image platform: {actual_platform}")

            if image_os != expected_os or image_arch != expected_arch:
                raise RuntimeError(
                    f"{image} is {actual_platform}, but the Recipe requests "
                    f"{expected_os}/{expected_arch}"
                )

            if is_pinned_pelita_image(image) and image_id != PELITA_IMAGE_ID:
                raise RuntimeError(
                    "The official Pelita-875 image ID does not match the recorded original pull. "
                    f"Expected {PELITA_IMAGE_ID}, got {image_id}. "
                    "Check the registry tag before running the experiment."
                )

            if daemon_os != image_os or daemon_arch != image_arch:
                raise RuntimeError(
                    f"Docker host {daemon_os}/{daemon_arch} differs from {actual_platform}. "
                    "Select a native image; QEMU E2E is not supported."
                )

            probe = run(
                [
                    "docker", "run", "--rm", "--pull", "never",
                    "--platform", platform_name,
                    "--entrypoint", "/bin/bash",
                    image, "-lc", "uname -m",
                ],
                capture=True,
            )
            if probe.returncode != 0:
                detail = (probe.stderr or probe.stdout).strip()
                raise RuntimeError(
                    f"Cannot execute {image} as {platform_name}. Docker error: {detail}"
                )

            container_arch = probe.stdout.strip().splitlines()[-1]
            expected_uname = {"amd64": "x86_64", "arm64": "aarch64"}.get(expected_arch)
            if expected_uname and container_arch != expected_uname:
                raise RuntimeError(
                    f"Image execution reported uname -m={container_arch!r}; "
                    f"expected {expected_uname!r}"
                )
            print(f"[image-check] PASS: runnable as {platform_name}; uname -m={container_arch}")

        print("\n[image-check] PASS: task images map correctly and run natively.")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[image-check] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
