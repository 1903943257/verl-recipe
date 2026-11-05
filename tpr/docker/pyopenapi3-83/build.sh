#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
IMAGE="${IMAGE:-tpr-pyopenapi3-83-claude:2.1.308}"

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker CLI not found" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon unavailable" >&2; exit 1; }

docker build --pull --platform linux/amd64 \
  --file "$SCRIPT_DIR/Dockerfile" \
  --tag "$IMAGE" \
  "$SCRIPT_DIR"

ARCH="$(docker image inspect "$IMAGE" --format '{{.Os}}/{{.Architecture}}')"
[[ "$ARCH" == "linux/amd64" ]] || { echo "ERROR: expected linux/amd64, got $ARCH" >&2; exit 1; }

docker run --rm --platform linux/amd64 --entrypoint /bin/bash "$IMAGE" -lc '
set -euo pipefail
test "$(uname -m)" = "x86_64"
test -f /opt/miniconda3/bin/activate
source /opt/miniconda3/bin/activate
conda activate testbed
python --version
python -m pip check
python -c "from pyopenapi3.schemas import ObjectsDTSchema; print(ObjectsDTSchema)"
python -m pytest --version
claude --version
test "$(git -C /testbed rev-parse HEAD)" = "2ef34c3213eb292703e0e5e6f2185b1e4725bbde"
'
echo "BUILD_PASS image=$IMAGE arch=$ARCH"
