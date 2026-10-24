#!/usr/bin/env python

"""Record a NexArm collection session and merge it into a root dataset.

This wraps ``lerobot-record`` (:func:`lerobot.scripts.lerobot_record.record`). It accepts every
``lerobot-record`` flag plus the NexArm collection flags:

- ``--root_repo_id`` / ``--merge_root``: merge the finished session into this local root dataset,
  keeping its earlier episodes. Compatibility (fps, robot type, features incl. cameras and
  resolution) is checked before recording; a root that exists neither locally nor on the Hub is
  created by the first session. The session itself is not pushed to the Hub.
- ``--manual_control=true`` / ``--recording_log``: wait for Enter/Right, record until pressed again,
  reset without a timer, and show a small terminal dashboard while routine logs go to the log file.
- ``--rerun_save_path``: also save the Rerun stream to a local ``.rrd`` file (needs
  ``--display_data=true``).

Example (``tools/nexarm/prepare_collection.py`` builds this command for you)::

    uv run python examples/nexarm/record_collection.py \\
        --robot.type=nexarm_follower --robot.port=/dev/ttyUSB1 \\
        --teleop.type=nexarm_leader --teleop.port=/dev/ttyUSB0 \\
        --dataset.repo_id=thanhkt/nexarm_stack_bowls_session \\
        --dataset.single_task="Stack the bowls" --dataset.num_episodes=10 \\
        --dataset.episode_time_s=inf --dataset.reset_time_s=inf \\
        --manual_control=true --root_repo_id=thanhkt/nexarm_stack_bowls_top
"""

import functools
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import RobotProcessorPipeline, make_default_processors
from lerobot.robots import Robot, make_robot_from_config
from lerobot.scripts import lerobot_record
from lerobot.scripts.lerobot_record import RecordConfig, record, record_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import shutdown_visualization
from lerobot_robot_nexarm.recording_session import (
    init_manual_keyboard_listener,
    run_manual_session,
    show_recording_status,
)
from lerobot_robot_nexarm.rerun_recording import init_visualization


@dataclass
class CollectionRecordConfig(RecordConfig):
    # Optional local .rrd file for a replayable Rerun recording of this session.
    # Requires --display_data=true and --display_mode=rerun.
    rerun_save_path: str | None = None
    # Wait for Enter/Right, record until pressed again, and reset without a timer.
    manual_control: bool = False
    # Write routine logs here while showing the manual collection dashboard.
    recording_log: str = "outputs/collection/recording.log"
    # Merge finished sessions into this local root dataset, preserving prior episodes. Compatibility
    # (fps, robot type, features incl. cameras and resolution) is checked before recording; a root
    # that exists neither locally nor on the Hub is created by the first session.
    root_repo_id: str | None = None
    merge_root: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.rerun_save_path is not None and not self.display_data:
            raise ValueError("--rerun_save_path requires --display_data=true.")


Processors = tuple[RobotProcessorPipeline, RobotProcessorPipeline, RobotProcessorPipeline]


def _dataset_features(cfg: RecordConfig, robot: Robot, processors: Processors) -> dict:
    """The dataset features ``lerobot-record`` derives for this robot and these processors."""
    teleop_action_processor, _, robot_observation_processor = processors
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )


def _merge_root(cfg: CollectionRecordConfig) -> Path | None:
    return Path(cfg.merge_root) if cfg.merge_root else None


def _check_mergeable(cfg: CollectionRecordConfig, robot: Robot, features: dict) -> None:
    from lerobot_robot_nexarm.collection_merge import check_session_mergeable

    check_session_mergeable(
        cfg.root_repo_id,
        _merge_root(cfg),
        fps=cfg.dataset.fps,
        robot_type=robot.robot_type,
        features=features,
    )


@contextmanager
def _rerun_saved_to(path: str | None):
    """Make ``lerobot-record`` pass ``rerun_save_path`` when it starts the visualization."""
    if path is None:
        yield
        return
    original = lerobot_record.init_visualization
    lerobot_record.init_visualization = functools.partial(init_visualization, rerun_save_path=path)
    try:
        yield
    finally:
        lerobot_record.init_visualization = original


def _open_dataset(cfg: RecordConfig, robot: Robot, features: dict) -> LeRobotDataset:
    """Create or resume the session dataset exactly as ``lerobot-record`` does."""
    num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
    if cfg.resume:
        dataset = LeRobotDataset.resume(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            rgb_encoder=cfg.dataset.rgb_encoder,
            depth_encoder=cfg.dataset.depth_encoder,
            encoder_threads=cfg.dataset.encoder_threads,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, features)
        return dataset
    if cfg.dataset.repo_id.split("/", 1)[-1].startswith("eval_"):
        raise ValueError(
            "Dataset names starting with 'eval_' are reserved for policy evaluation. "
            "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
        )
    cfg.dataset.stamp_repo_id()
    return LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.robot_type,
        features=features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        rgb_encoder=cfg.dataset.rgb_encoder,
        depth_encoder=cfg.dataset.depth_encoder,
        encoder_threads=cfg.dataset.encoder_threads,
        streaming_encoding=cfg.dataset.streaming_encoding,
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
    )


def _log_to_file(path: str) -> None:
    """Send routine logs to *path* so they do not scroll the manual dashboard away."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    for old_handler in root_logger.handlers[:]:
        root_logger.removeHandler(old_handler)
        old_handler.close()
    root_logger.addHandler(handler)


def _record_manual(cfg: CollectionRecordConfig, processors: Processors) -> LeRobotDataset | None:
    """Record episodes between deliberate key presses (``--manual_control=true``)."""
    init_logging()
    _log_to_file(cfg.recording_log)
    cfg.play_sounds = False
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_visualization(
            cfg.display_mode,
            session_name="recording",
            ip=cfg.display_ip,
            port=cfg.display_port,
            rerun_save_path=cfg.rerun_save_path,
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    teleop_action_processor, robot_action_processor, robot_observation_processor = processors
    features = _dataset_features(cfg, robot, processors)
    if cfg.root_repo_id:
        _check_mergeable(cfg, robot, features)

    dataset = None
    listener = None
    try:
        dataset = _open_dataset(cfg, robot, features)
        if teleop is not None:
            teleop.connect()
        robot.connect()

        listener, events = init_manual_keyboard_listener()
        if listener is None:
            raise RuntimeError("Manual collection requires an interactive terminal for Enter/Right controls.")

        def run_phase(phase_dataset: LeRobotDataset | None) -> None:
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=phase_dataset,
                control_time_s=float("inf"),
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
                display_mode=cfg.display_mode,
                display_compressed_images=display_compressed_images,
            )

        with VideoEncodingManager(dataset):
            run_manual_session(dataset, events, cfg.dataset.num_episodes, run_phase)
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        if dataset:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)
        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")
        log_say("Exiting", cfg.play_sounds)
    return dataset


def _merge(cfg: CollectionRecordConfig, dataset: LeRobotDataset) -> None:
    from lerobot_robot_nexarm.collection_merge import merge_collection_session

    if cfg.manual_control:
        show_recording_status(dataset.num_episodes, "MERGING DATA...", cfg.root_repo_id)
    try:
        merge_collection_session(dataset, cfg.root_repo_id, _merge_root(cfg))
    except Exception:
        logging.exception("Local merge failed. Local session retained at %s", dataset.root)
        print(
            "Merge failed; local episodes are saved. Retry with:\n"
            f"uv run --no-sync python tools/nexarm/merge_collection.py --repo-id {dataset.repo_id} "
            f"--root '{dataset.root}' --root-repo-id {cfg.root_repo_id}"
            + (f" --merge-root '{cfg.merge_root}'" if cfg.merge_root else ""),
            flush=True,
        )
        raise


@parser.wrap()
def record_collection(cfg: CollectionRecordConfig) -> LeRobotDataset | None:
    if cfg.root_repo_id:
        # The merged root is the dataset that gets shared; the session stays local.
        cfg.dataset.push_to_hub = False
    processors = make_default_processors()

    if cfg.manual_control:
        dataset = _record_manual(cfg, processors)
    else:
        if cfg.root_repo_id:
            # Fail before any arm motion when the session could not be merged afterwards.
            robot = make_robot_from_config(cfg.robot)
            _check_mergeable(cfg, robot, _dataset_features(cfg, robot, processors))
        with _rerun_saved_to(cfg.rerun_save_path):
            # `record` is CLI-wrapped and only accepts an exact RecordConfig; call the function itself.
            dataset = record.__wrapped__(cfg, *processors)

    if cfg.root_repo_id and dataset is not None and dataset.num_episodes > 0:
        _merge(cfg, dataset)
    if cfg.manual_control and dataset is not None:
        show_recording_status(dataset.num_episodes, "FINISHED", str(dataset.root))
    if dataset is not None:
        # New sessions get a timestamp suffix; follow-up commands (replay, train) need this exact id.
        print(f"Dataset repo_id: {dataset.repo_id}\nDataset root: {dataset.root}", flush=True)
    return dataset


def main() -> None:
    register_third_party_plugins()
    record_collection()


if __name__ == "__main__":
    main()
