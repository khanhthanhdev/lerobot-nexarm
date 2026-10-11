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


@pytest.mark.parametrize(
    "flag,value",
    [("--batch-size", "0"), ("--grad-accum-steps", "0"), ("--ema-decay", "1"), ("--warmup-steps", "-1")],
)
def test_invalid_training_options(monkeypatch, flag, value):
    with pytest.raises(SystemExit):
        parse_args(monkeypatch, flag, value)


def test_collator_preserves_padding_and_image_dimensions():
    import torch

    stats = {key: {"min": [0.0] * 6, "max": [1.0] * 6} for key in ("action", "observation.state")}
    processor = Mock(return_value={"pixel_values": torch.zeros(1, 3, 32, 32)})
    collator = train_turbovla.TurboVLANexArmCollator(["front"], stats, processor, horizon=3)
    batch = collator(
        [
            {
                "task": "stack bowls",
                "front": torch.zeros(3, 48, 64),
                "observation.state": torch.zeros(6),
                "action": torch.ones(3, 6),
                "action_is_pad": torch.tensor([False, True, True]),
            }
        ]
    )
    assert batch["action_is_pad"].tolist() == [[False, True, True]]
    assert processor.call_args.kwargs["images"][0].size == (64, 48)
    assert torch.all(batch["actions"] == 1)


def test_small_distributed_mix_has_samples_on_every_rank():
    for rank in range(4):
        sampler = train_turbovla.DomainMixSampler([1, 1], [0.5, 0.5], num_replicas=4, rank=rank)
        assert len(sampler) == 1
        assert len(list(sampler)) == 1


def test_training_counts_optimizer_updates(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    import torch

    from lerobot.datasets import lerobot_dataset

    root = make_local_root(tmp_path / "data")
    parse_args(
        monkeypatch,
        "--dataset-root",
        str(root),
        "--output-dir",
        str(tmp_path / "out"),
        "--device",
        "cpu",
        "--max-steps",
        "2",
        "--warmup-steps",
        "0",
        "--batch-size",
        "4",
        "--grad-accum-steps",
        "3",
        "--num-workers",
        "0",
        "--horizon",
        "2",
        "--ema-decay",
        "0",
    )
    features = {key: {"shape": [6]} for key in ("action", "observation.state")}
    features["front"] = {"dtype": "image", "shape": [8, 8, 3]}
    stats = {key: {"min": [0.0] * 6, "max": [1.0] * 6} for key in ("action", "observation.state")}
    meta = SimpleNamespace(fps=30, stats=stats)

    class Dataset(torch.utils.data.Dataset):
        num_episodes = 1

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {
                "task": "stack",
                "front": torch.zeros(3, 8, 8),
                "observation.state": torch.zeros(6),
                "action": torch.ones(2, 6),
                "action_is_pad": torch.tensor([False, True]),
            }

    ds = Dataset()
    ds.meta, ds.features = meta, features
    monkeypatch.setattr(lerobot_dataset, "LeRobotDatasetMetadata", Mock(return_value=meta))
    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", Mock(return_value=ds))
    processor = Mock(return_value={"pixel_values": torch.zeros(1, 3, 8, 8)})
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoImageProcessor=SimpleNamespace(from_pretrained=lambda _: processor)),
    )
    config = SimpleNamespace(
        vision=SimpleNamespace(),
        text=SimpleNamespace(),
        action=SimpleNamespace(),
        interaction=SimpleNamespace(),
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = torch.nn.Linear(6, 6)
            self.text_encoder = torch.nn.Linear(1, 1)
            self.vision_encoder = torch.nn.Linear(1, 1)
            self.vision_language_interaction = torch.nn.Linear(1, 1)
            self.calls = 0

        def forward(self, instructions, samples, states):
            self.calls += 1
            assert not self.text_encoder.training
            return self.action_head(states).unsqueeze(1).expand(-1, 2, -1)

    model = Model()
    monkeypatch.setitem(
        sys.modules,
        "turbovla.models",
        SimpleNamespace(TurboVLAConfig=lambda: config, build_turbovla=lambda _: model),
    )
    save = Mock()
    monkeypatch.setattr(train_turbovla, "save_checkpoint", save)
    train_turbovla.main()
    assert model.calls == 6
    assert save.call_args.args[-1] == 2
    assert config.text.frozen


def test_camera_hints_only_match_images_and_honor_legacy_names():
    features = {
        "front_state": {"dtype": "float32"},
        "observation.images.front": {"dtype": "video"},
        "observation.images.wrist": {"dtype": "video"},
    }
    assert train_turbovla.resolve_camera_keys(features, "front") == ["observation.images.front"]
    with pytest.raises(ValueError, match="exactly one"):
        train_turbovla.resolve_camera_keys(features, "front,missing")
    with pytest.raises(ValueError, match="Duplicate"):
        train_turbovla.resolve_camera_keys(features, "front,front")


def test_legacy_camera_key_flags_select_front_and_wrist():
    features = {
        "observation.images.front": {"dtype": "video"},
        "observation.images.wrist": {"dtype": "video"},
        "observation.images.top": {"dtype": "video"},
        "observation.images.side": {"dtype": "video"},
    }
    # The unset key is matched by name; the model always gets [front, wrist] in that order.
    assert train_turbovla.resolve_camera_keys(features, wrist_hint="observation.images.wrist") == [
        "observation.images.front",
        "observation.images.wrist",
    ]
    assert train_turbovla.resolve_camera_keys(features, front_hint="side") == [
        "observation.images.side",
        "observation.images.wrist",
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        train_turbovla.resolve_camera_keys(features, front_hint="observation.images.wrist")


def test_cameras_and_legacy_camera_key_flags_conflict(monkeypatch):
    with pytest.raises(SystemExit):
        parse_args(monkeypatch, "--cameras", "front,wrist", "--wrist-cam-key", "wrist")


def tasks_meta(fps, *tasks):
    from types import SimpleNamespace

    import pandas as pd

    index = pd.Index(tasks, name="task")
    return SimpleNamespace(fps=fps, tasks=pd.DataFrame({"task_index": range(len(tasks))}, index=index))


def test_checkpoint_metadata_prefers_real_prompt_and_infers_task_type():
    sim = tasks_meta(30, "Stack the bowls with blue on bottom, red in middle, and black on top.")
    real = tasks_meta(30, "Stack the bowls with red on bottom, blue in middle, and black on top.")
    keys = ["observation.images.front", "observation.images.wrist"]

    metadata = train_turbovla.checkpoint_metadata(["sim", "real"], [sim, real], keys, "auto")

    assert metadata == {
        "task_type": "stack_bowls",
        "camera_keys": keys,
        "fps": 30,
        "task": "Stack the bowls with red on bottom, blue in middle, and black on top.",
    }
    pick = tasks_meta(30, "Pick up the red cube, place it in the green target zone, and release it.")
    assert train_turbovla.checkpoint_metadata(["sim"], [pick], keys, "auto")["task_type"] == "pick_place"
    explicit = train_turbovla.checkpoint_metadata(["sim"], [pick], keys, "stack_bowls")
    assert explicit["task_type"] == "stack_bowls"


def test_checkpoint_metadata_requires_one_fps():
    metas = [tasks_meta(30, "a"), tasks_meta(15, "b")]
    with pytest.raises(ValueError, match="share one fps"):
        train_turbovla.checkpoint_metadata(["sim", "real"], metas, [], "auto")


def test_saved_config_round_trips_through_rollout_loader(tmp_path):
    import json
    from types import SimpleNamespace

    import torch

    from examples.nexarm.rollout_turbovla import load_checkpoint_config

    config = SimpleNamespace(
        action=SimpleNamespace(action_dim=6, state_dim=6, horizon=8),
        vision=SimpleNamespace(num_views=3, image_size=256),
    )
    stats = {key: {"min": [0.0] * 6, "max": [1.0] * 6} for key in ("action", "observation.state")}
    metadata = {
        "task_type": "stack_bowls",
        "camera_keys": ["observation.images.top", "observation.images.front", "observation.images.wrist"],
        "fps": 15,
        "task": "Stack the bowls with red on bottom, blue in middle, and black on top.",
    }
    train_turbovla.save_checkpoint(
        torch.nn.Linear(1, 1), None, config, stats, tmp_path, 2, is_final=True, metadata=metadata
    )

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["vision"] == {"num_views": 3, "image_size": 256}
    loaded = load_checkpoint_config(tmp_path / "final_model.pt")
    assert loaded.camera_names == ["top", "front", "wrist"]
    assert (loaded.image_size, loaded.horizon, loaded.fps, loaded.task_type) == (256, 8, 15, "stack_bowls")
    assert loaded.task == metadata["task"]
