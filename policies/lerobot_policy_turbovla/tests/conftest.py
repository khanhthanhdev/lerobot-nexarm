"""Shared fixtures: the locally cached NexArm dataset and small CPU backbones for TurboVLA."""

import os
from pathlib import Path

import pytest

TEXT_WEIGHTS = "prajjwal1/bert-tiny"  # has no tokenizer files; bert-base-uncased's tokenizer is added
TOKENIZER = "google-bert/bert-base-uncased"
VISION_MODEL = "facebook/dinov2-with-registers-small"


@pytest.fixture(scope="session")
def nexarm_dataset_root() -> Path:
    """Root of the cached `thanhkt/nexarm_stack_bowls` snapshot (or `TURBOVLA_PARITY_DATASET_ROOT`)."""
    if root := os.environ.get("TURBOVLA_PARITY_DATASET_ROOT"):
        return Path(root)
    from lerobot.utils.constants import HF_LEROBOT_HOME

    snapshots = sorted(
        (HF_LEROBOT_HOME / "hub" / "datasets--thanhkt--nexarm_stack_bowls" / "snapshots").glob("*")
    )
    for snapshot in snapshots:
        if (snapshot / "meta" / "info.json").is_file():
            return snapshot
    pytest.skip("thanhkt/nexarm_stack_bowls is not in the local cache; set TURBOVLA_PARITY_DATASET_ROOT")


@pytest.fixture(scope="session")
def tiny_backbones(tmp_path_factory) -> tuple[str, str]:
    """(text model folder, vision model id): BERT-tiny with a BERT tokenizer and DINOv2-small."""
    transformers = pytest.importorskip("transformers")
    text_dir = tmp_path_factory.mktemp("bert_tiny")
    try:
        transformers.BertModel.from_pretrained(TEXT_WEIGHTS).save_pretrained(text_dir)
        transformers.AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(text_dir)
        transformers.AutoImageProcessor.from_pretrained(VISION_MODEL)
        transformers.AutoModel.from_pretrained(VISION_MODEL)
    except OSError as error:
        pytest.skip(f"Backbones are neither cached nor downloadable: {error}")
    return str(text_dir), VISION_MODEL
