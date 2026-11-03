from __future__ import annotations

import pytest
import torch

from verl_tpr.tree import build_tree_execution_plans


MIN_SHARED = 1024


def _build(
    rows: list[list[int]],
    masks: list[list[int]],
    uids: list[str],
    *,
    min_shared: int = MIN_SHARED,
    include_responses: bool = False,
    loss_masks: list[list[int]] | None = None,
    require_loss_mask_alignment: bool = False,
):
    keys = [f"{uid}_session_{index}" for index, uid in enumerate(uids)]
    batch = {
        "input_ids": [torch.tensor(row, dtype=torch.long) for row in rows],
        "response_mask": [torch.tensor(mask, dtype=torch.long) for mask in masks],
    }
    if include_responses:
        batch["responses"] = [
            torch.tensor(row[-len(mask) :], dtype=torch.long)
            for row, mask in zip(rows, masks, strict=True)
        ]
    if loss_masks is not None:
        batch["loss_mask"] = [
            torch.tensor(mask, dtype=torch.long) for mask in loss_masks
        ]
    return build_tree_execution_plans(
        keys,
        batch,
        min_shared_prefix_tokens=min_shared,
        require_loss_mask_alignment=require_loss_mask_alignment,
    )


def _assert_objective_mapping(
    forest,
    rows: list[list[int]],
    masks: list[list[int]],
) -> None:
    expected = {
        (row_index, response_offset)
        for row_index, mask in enumerate(masks)
        for response_offset, selected in enumerate(mask)
        if selected
    }
    actual = []

    for tree in forest.trees:
        plan = tree.segment_plan
        for segment in plan.segments.values():
            start, end = segment.position_start, segment.position_end
            assert any(
                rows[row][start:end] == segment.token_ids.tolist()
                for row in tree.member_rows
            )

        for ref in tree.objective_refs:
            row = ref.sample_row
            assert row in tree.member_rows
            response_start = len(rows[row]) - len(masks[row])
            target_position = response_start + ref.response_offset
            query_position = target_position - 1
            segment = plan.get(ref.segment_id)

            assert segment.position_start <= query_position < segment.position_end
            assert ref.query_offset == query_position - segment.position_start
            assert int(segment.token_ids[ref.query_offset]) == rows[row][query_position]
            assert ref.target_token_id == rows[row][target_position]
            actual.append((row, ref.response_offset))

    assert sorted(actual) == sorted(expected)
    assert len(actual) == len(set(actual))
    assert forest.logical_loss_tokens == len(expected)


def test_shared_prefix_at_threshold_builds_one_tree_across_uids() -> None:
    prefix = list(range(10_000, 10_000 + 1100))
    suffix_a = list(range(90_000, 90_024))
    suffix_b = list(range(80_000, 80_024))
    rows = [prefix + suffix_a, prefix + suffix_b]
    masks = [
        [int(index % 3 != 1) for index in range(24)],
        [int(index % 4 != 0) for index in range(24)],
    ]
    loss_masks = [
        [0] * len(prefix) + masks[0],
        masks[1],
    ]

    forest = _build(
        rows,
        masks,
        ["uid-a", "uid-b"],
        include_responses=True,
        loss_masks=loss_masks,
        require_loss_mask_alignment=True,
    )

    assert len(forest.trees) == 1
    tree = forest.trees[0]
    assert tree.member_rows == (0, 1)
    assert tree.member_uids == ("uid-a", "uid-b")

    plan = tree.segment_plan
    root = plan.get(0)
    assert root.parent_id is None
    assert root.token_ids.tolist() == prefix
    children = plan.children_of(root.segment_id)
    assert len(children) == 2
    assert sorted(child.length for child in children) == [24, 24]

    _assert_objective_mapping(forest, rows, masks)


def test_prefix_below_threshold_is_not_shared() -> None:
    prefix = list(range(20_000, 20_000 + MIN_SHARED - 1))
    rows = [
        prefix + list(range(30_000, 30_012)),
        prefix + list(range(40_000, 40_012)),
    ]
    masks = [[1] * 12, [1] * 12]

    forest = _build(rows, masks, ["uid-a", "uid-b"])

    assert len(forest.trees) == 2
    assert {row for tree in forest.trees for row in tree.member_rows} == {0, 1}
    assert all(len(tree.member_rows) == 1 for tree in forest.trees)
    assert all(len(tree.segment_plan.segments) == 1 for tree in forest.trees)
    _assert_objective_mapping(forest, rows, masks)


def test_short_internal_shared_branch_is_expanded_into_leaf_segments() -> None:
    prefix = list(range(10_000, 10_000 + 1100))
    short_branch = [71, 72, 73]
    rows = [
        prefix,
        prefix + short_branch + list(range(30_000, 30_012)),
        prefix + short_branch + list(range(40_000, 40_012)),
    ]
    masks = [[1] * 8, [1] * 8, [1] * 8]

    forest = _build(rows, masks, ["uid-a", "uid-b", "uid-c"])

    assert len(forest.trees) == 1
    tree = forest.trees[0]
    assert tree.member_rows == (0, 1, 2)

    plan = tree.segment_plan
    root = plan.get(0)
    assert root.token_ids.tolist() == prefix
    children = plan.children_of(root.segment_id)
    assert len(children) == 2
    assert all(child.position_start == len(prefix) for child in children)
    assert all(child.length == len(short_branch) + 12 for child in children)
    assert all(child.token_ids.tolist()[:3] == short_branch for child in children)

    _assert_objective_mapping(forest, rows, masks)


def test_long_nested_shared_branch_is_retained_as_a_segment() -> None:
    prefix = list(range(10_000, 10_000 + 1100))
    shared_branch = list(range(20_000, 20_000 + 1050))
    rows = [
        prefix + list(range(15_000, 15_012)),
        prefix + shared_branch + list(range(30_000, 30_012)),
        prefix + shared_branch + list(range(40_000, 40_012)),
    ]
    masks = [[1] * 8, [1] * 8, [1] * 8]

    forest = _build(rows, masks, ["uid-a", "uid-b", "uid-c"])

    assert len(forest.trees) == 1
    tree = forest.trees[0]
    assert tree.member_rows == (0, 1, 2)

    plan = tree.segment_plan
    root = plan.get(0)
    assert root.token_ids.tolist() == prefix
    root_children = plan.children_of(root.segment_id)
    assert sorted(child.length for child in root_children) == [12, 1050]

    shared_segment = next(child for child in root_children if child.length == 1050)
    assert shared_segment.token_ids.tolist() == shared_branch
    assert len(plan.children_of(shared_segment.segment_id)) == 2

    _assert_objective_mapping(forest, rows, masks)


def test_identical_rows_from_different_uids_share_one_segment() -> None:
    row = list(range(50_000, 50_000 + 1100)) + list(range(60_000, 60_016))
    rows = [row, row.copy()]
    masks = [[1] * 8, [1] * 8]

    forest = _build(rows, masks, ["uid-a", "uid-b"])

    assert len(forest.trees) == 1
    tree = forest.trees[0]
    assert tree.member_rows == (0, 1)
    assert tree.member_uids == ("uid-a", "uid-b")
    assert len(tree.segment_plan.segments) == 1
    assert tree.segment_plan.get(0).token_ids.tolist() == row

    _assert_objective_mapping(forest, rows, masks)


def test_loss_mask_mismatch_is_rejected() -> None:
    row = list(range(100))
    mask = [1] * 8

    with pytest.raises(ValueError, match="response_mask and loss_mask disagree"):
        _build(
            [row],
            [mask],
            ["uid-a"],
            loss_masks=[[0] * len(mask)],
            require_loss_mask_alignment=True,
        )
