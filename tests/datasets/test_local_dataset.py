"""The local dataset helpers load a complete local root and name missing files without contacting the Hub."""

import shutil

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import lerobot.datasets.dataset_metadata as dataset_metadata_module  # noqa: E402
import lerobot.datasets.lerobot_dataset as lerobot_dataset_module  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot_robot_nexarm.local_dataset import load_local_dataset  # noqa: E402

REPO_ID = "local/nexarm_dataset"
FEATURES = {
    "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
}


@pytest.fixture
def no_hub(monkeypatch):
    def unexpected_hub_call(*args, **kwargs):
        pytest.fail("A local-only dataset must not contact the Hub")

    for module in (dataset_metadata_module, lerobot_dataset_module):
        monkeypatch.setattr(module, "get_safe_version", unexpected_hub_call)
        monkeypatch.setattr(module, "snapshot_download", unexpected_hub_call)


def make_dataset(root):
    dataset = LeRobotDataset.create(repo_id=REPO_ID, root=root, fps=10, features=FEATURES, use_videos=False)
    for _ in range(3):
        dataset.add_frame(
            {
                "action": np.zeros(2, dtype=np.float32),
                "observation.state": np.zeros(2, dtype=np.float32),
                "task": "Dummy task",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return dataset


@pytest.mark.parametrize("missing", [None, "meta/info.json", "meta/tasks.parquet", "meta/episodes", "data"])
def test_local_dataset_never_contacts_hub(tmp_path, no_hub, missing):
    root = tmp_path / "local"
    make_dataset(root)
    if missing is not None:
        path = root / missing
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    if missing is None:
        assert len(load_local_dataset(REPO_ID, root)) == 3
    else:
        with pytest.raises(FileNotFoundError, match="Hub downloads are disabled") as exc:
            load_local_dataset(REPO_ID, root)
        assert str(root) in str(exc.value)


def test_local_dataset_reports_missing_video(tmp_path, no_hub):
    camera = {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channels"]}
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=tmp_path / "local_video",
        fps=10,
        features={**FEATURES, "observation.images.front": camera},
        use_videos=True,
    )
    for _ in range(3):
        dataset.add_frame(
            {
                "action": np.zeros(2, dtype=np.float32),
                "observation.state": np.zeros(2, dtype=np.float32),
                "observation.images.front": np.zeros((64, 64, 3), dtype=np.uint8),
                "task": "Dummy task",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    video = dataset.root / dataset.meta.get_video_file_path(0, "observation.images.front")
    video.unlink()

    with pytest.raises(FileNotFoundError, match="episode data is incomplete") as exc:
        load_local_dataset(REPO_ID, dataset.root)
    assert str(video) in str(exc.value)
