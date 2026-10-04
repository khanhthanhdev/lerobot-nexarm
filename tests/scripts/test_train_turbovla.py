# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from unittest.mock import Mock

import pytest

from examples.nexarm import train_turbovla


def parse_args(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["train_turbovla.py", *argv])
    return train_turbovla.parse_args()


def make_local_root(root):
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    return root


def test_local_sim_and_hub_real(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    root = make_local_root(tmp_path / "datasets" / "sim")
    args = parse_args(
        monkeypatch,
        "--sim-dataset-root",
        "datasets/sim",
        "--real-repo-id",
        "thanhkt/nexarm_stack_bowls",
        "--real-ratio",
        "0.5",
    )

    assert train_turbovla.resolve_dataset_specs(args) == [
        ("sim", "local/nexarm_dataset", root),
        ("real", "thanhkt/nexarm_stack_bowls", None),
    ]


@pytest.mark.parametrize("root_kind", ["missing", "empty", "parent", "file"])
def test_invalid_local_root_fails_before_distributed_or_hub_setup(monkeypatch, tmp_path, root_kind):
    root = tmp_path / "sim"
    if root_kind == "empty":
        root.mkdir()
    elif root_kind == "parent":
        make_local_root(root / "nested_dataset")
    elif root_kind == "file":
        root.write_text("not a dataset")
    parse_args(
        monkeypatch,
        "--sim-dataset-root",
        str(root),
        "--real-repo-id",
        "thanhkt/nexarm_stack_bowls",
        "--real-ratio",
        "0.5",
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    init = Mock()
    monkeypatch.setattr(train_turbovla.torch.distributed, "init_process_group", init)
    monkeypatch.setattr(train_turbovla.torch.cuda, "set_device", Mock())

    with pytest.raises(FileNotFoundError, match="Local sim dataset metadata not found") as exc:
        train_turbovla.main()

    assert str(root / "meta" / "info.json") in str(exc.value)
    assert "--sim-dataset-root" in str(exc.value)
    init.assert_not_called()
    assert root.exists() == (root_kind != "missing")


def test_hub_sim_does_not_use_default_local_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = parse_args(monkeypatch, "--sim-repo-id", "owner/sim", "--real-repo-id", "owner/real")

    assert train_turbovla.resolve_dataset_specs(args) == [
        ("sim", "owner/sim", None),
        ("real", "owner/real", None),
    ]


def test_default_local_root_and_legacy_alias(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    root = make_local_root(tmp_path / train_turbovla.DEFAULT_SIM_DATASET_ROOT)
    default = train_turbovla.resolve_dataset_specs(parse_args(monkeypatch))
    legacy = train_turbovla.resolve_dataset_specs(
        parse_args(monkeypatch, "--dataset-root", str(root), "--repo-id", "local/sim")
    )

    assert default == [("sim", "local/nexarm_dataset", root)]
    assert legacy == [("sim", "local/sim", root)]


def test_real_only_hub_skips_sim_root(monkeypatch):
    args = parse_args(monkeypatch, "--no-sim", "--real-repo-id", "owner/real")

    assert train_turbovla.resolve_dataset_specs(args) == [("real", "owner/real", None)]


def test_explicit_local_real_root_requires_metadata_even_with_repo_id(monkeypatch, tmp_path):
    args = parse_args(
        monkeypatch,
        "--no-sim",
        "--real-dataset-root",
        str(tmp_path / "real"),
        "--real-repo-id",
        "owner/real",
    )

    with pytest.raises(FileNotFoundError, match="Local real dataset metadata not found"):
        train_turbovla.resolve_dataset_specs(args)


@pytest.mark.parametrize("ratio", ["0", "1", "-0.1", "1.1", "nan"])
def test_invalid_mix_ratio(monkeypatch, ratio):
    args = parse_args(
        monkeypatch, "--sim-repo-id", "owner/sim", "--real-repo-id", "owner/real", "--real-ratio", ratio
    )
    with pytest.raises(ValueError, match="--real-ratio"):
        train_turbovla.resolve_dataset_specs(args)


def test_mix_ratio_requires_two_datasets(monkeypatch):
    args = parse_args(monkeypatch, "--no-sim", "--real-repo-id", "owner/real", "--real-ratio", "0.5")
    with pytest.raises(ValueError, match="requires both"):
        train_turbovla.resolve_dataset_specs(args)


def test_no_datasets_selected(monkeypatch):
    args = parse_args(monkeypatch, "--no-sim")
    with pytest.raises(ValueError, match="No dataset selected"):
        train_turbovla.resolve_dataset_specs(args)
