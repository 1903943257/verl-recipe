# Keep the original SWE-ReBench task environment (including Conda and /testbed).
# Add the same pinned Claude Code release used by the Pelita AMD64 image.
FROM --platform=linux/amd64 node:22.14.0-bookworm-slim AS claude

RUN npm install --global --no-audit --no-fund @anthropic-ai/claude-code@2.1.308

FROM --platform=linux/amd64 swerebench/sweb.eval.x86_64.algebra8_1776_pyopenapi3-83:latest

ARG TASK_BASE_COMMIT=2ef34c3213eb292703e0e5e6f2185b1e4725bbde

COPY --from=claude /usr/local/bin/node /usr/local/bin/node
COPY --from=claude /usr/local/lib/node_modules /usr/local/lib/node_modules

ENV PATH="/usr/local/bin:${PATH}"
ENV DISABLE_AUTOUPDATER=1
ENV PYTHONUNBUFFERED=1

RUN ln -sfn /usr/local/lib/node_modules/@anthropic-ai/claude-code/cli.js /usr/local/bin/claude \
    && test -f /opt/miniconda3/bin/activate \
    && test -x /opt/miniconda3/envs/testbed/bin/python \
    && test -d /testbed/.git \
    && test "$(git -C /testbed rev-parse HEAD)" = "$TASK_BASE_COMMIT"

WORKDIR /testbed
