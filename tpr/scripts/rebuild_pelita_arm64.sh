#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="${IMAGE:-uniagent-swe-pelita-875:latest}"
BUILD_DIR="${BUILD_DIR:-/tmp/uniagent-swe-pelita-875-arm64}"
BUILD_LOG="${BUILD_LOG:-/tmp/uniagent-swe-pelita-875-arm64-build.log}"
MINICONDA_SHA256="5afee04df06c923462341cfc87727100a67fa854550d0d7a2e6295005168e6b7"
PELITA_COMMIT="68af15d8d4199882d32bb4ede363195e2c5b5a99"

die() { echo "ERROR: $*" >&2; exit 1; }
case "$(uname -m)" in aarch64|arm64) ;; *) die "Expected ARM64 host; got $(uname -m)" ;; esac
command -v docker >/dev/null || die "docker CLI not found"
docker info >/dev/null 2>&1 || die "Docker daemon unavailable"

# Read the Docker daemon proxy without printing credentials. Resolve its hostname
# on the host because the build container may not be able to resolve the proxy DNS.
PROXY_RAW="$(docker info 2>/dev/null | sed -n 's/^[[:space:]]*HTTPS Proxy: //p' | head -1)"
[[ -n "$PROXY_RAW" && "$PROXY_RAW" != "<no value>" ]] || \
  PROXY_RAW="$(docker info 2>/dev/null | sed -n 's/^[[:space:]]*HTTP Proxy: //p' | head -1)"
[[ -n "$PROXY_RAW" && "$PROXY_RAW" != "<no value>" ]] || die "No Docker daemon proxy found"
PROXY_BUILD="$(python3 - "$PROXY_RAW" <<'PY'
import socket, sys
from urllib.parse import urlsplit, urlunsplit

u = urlsplit(sys.argv[1])
if not u.scheme or not u.hostname:
    raise SystemExit("Cannot parse proxy URL")
ip = socket.gethostbyname(u.hostname)
userinfo = u.netloc.rsplit("@", 1)[0] + "@" if "@" in u.netloc else ""
if ":" in ip and not ip.startswith("["):
    ip = f"[{ip}]"
port = f":{u.port}" if u.port else ""
print(urlunsplit((u.scheme, userinfo + ip + port, u.path, u.query, u.fragment)))
PY
)" || die "Could not resolve proxy host"

mkdir -p "$BUILD_DIR"
cat > "$BUILD_DIR/Dockerfile" <<DOCKERFILE
FROM --platform=linux/arm64 node:22-bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/bin:\${PATH}
ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates curl git gcc g++ make ripgrep procps file && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py39_25.1.1-2-Linux-aarch64.sh -o /tmp/miniconda.sh && \
    echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum -c - && \
    bash /tmp/miniconda.sh -b -p /opt/conda && rm -f /tmp/miniconda.sh && ln -s /opt/conda /opt/miniconda3
RUN conda create -y -n testbed python=3.9.25 pip && conda clean -afy
RUN npm install --global @anthropic-ai/claude-code@2.1.161
RUN git clone https://github.com/ASPP/pelita.git /testbed && cd /testbed && git checkout ${PELITA_COMMIT} && \
    /opt/conda/envs/testbed/bin/python -m pip install --upgrade pip setuptools wheel && \
    /opt/conda/envs/testbed/bin/python -m pip install -e . pytest pytest-cov pytest-timestamper
WORKDIR /testbed
DOCKERFILE

echo "ARM64 host: $(uname -m)"
echo "Image: $IMAGE"
echo "Build context: $BUILD_DIR"
echo "Build log: $BUILD_LOG"
echo "Proxy detected and hostname resolved; credentials hidden."
if docker build --network=host \
  --build-arg HTTP_PROXY="$PROXY_BUILD" --build-arg HTTPS_PROXY="$PROXY_BUILD" \
  --build-arg http_proxy="$PROXY_BUILD" --build-arg https_proxy="$PROXY_BUILD" \
  -t "$IMAGE" "$BUILD_DIR" 2>&1 | tee "$BUILD_LOG"; then
  echo "========== Image architecture =========="
  docker image inspect "$IMAGE" --format '{{.Id}} {{.Os}}/{{.Architecture}}'
  echo "========== Runtime smoke check =========="
  docker run --rm --pull never --platform linux/arm64 "$IMAGE" bash -lc \
    'uname -m; /opt/conda/envs/testbed/bin/python --version; claude --version; git -C /testbed rev-parse HEAD; git -C /testbed cat-file -t '"$PELITA_COMMIT"
  echo "BUILD_PASS image=$IMAGE"
else
  rc=$?
  echo "BUILD_FAILED exit=$rc log=$BUILD_LOG context=$BUILD_DIR" >&2
  exit "$rc"
fi
