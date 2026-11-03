# Temporary Token-LCP tree validation

This directory is a temporary CPU-only regression suite for the compact Token-LCP
builder in `tpr/verl_tpr/tree.py`. It is not part of the final Recipe and can be
removed after the tree refactor is accepted.

Run from the repository root:

```bash
export PYTHONPATH="$PWD/tpr${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tpr/tests_tree_validation
```

The tests import only `verl_tpr.tree` and use CPU tensors. They do not require
VERL, Megatron, Ray, an NPU, or an E2E rollout.
