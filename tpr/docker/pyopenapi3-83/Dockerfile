# Keep the original SWE-ReBench task environment (including Conda and /testbed).
# Install the instance's declared project/test dependencies and add pinned Claude Code.
FROM node:22.14.0-bookworm-slim AS claude

RUN npm install --global --no-audit --no-fund @anthropic-ai/claude-code@2.1.308 \
    && test -x /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe

FROM swerebench/sweb.eval.x86_64.algebra8_1776_pyopenapi3-83:latest

ARG TASK_BASE_COMMIT=2ef34c3213eb292703e0e5e6f2185b1e4725bbde

# Claude's npm installer places the native Linux executable at bin/claude.exe.
# Copy only that binary; Node and npm are not needed at runtime.
COPY --from=claude /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe /usr/local/bin/claude

ENV DISABLE_AUTOUPDATER=1
ENV PYTHONUNBUFFERED=1

RUN chmod +x /usr/local/bin/claude \
    && test -f /opt/miniconda3/bin/activate \
    && test -x /opt/miniconda3/envs/testbed/bin/python \
    && test -d /testbed/.git \
    && test "$(git -C /testbed rev-parse HEAD)" = "$TASK_BASE_COMMIT"

# Match the SWE-ReBench install_config: pip install -e .[dev], plus pytest 6.2.2.
RUN /bin/bash -lc 'source /opt/miniconda3/bin/activate && conda activate testbed && cd /testbed && python -m pip install -e ".[dev]" "pytest==6.2.2"'

WORKDIR /testbed
