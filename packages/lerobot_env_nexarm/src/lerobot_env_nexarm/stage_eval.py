"""``lerobot-nexarm-eval``: ``lerobot-eval`` that also reports how far NexArm episodes got.

NexArm envs latch ``info["stage/<name>"]`` flags (the stack-bowls task reports one per stage). Under this
wrapper each env logs the final flags of every seeded episode, and the results gain ``pc_<stage>``, the
percentage of episodes that reached each stage, next to ``pc_success`` in ``overall``, ``per_group`` and
each ``per_task`` entry; ``per_task`` also lists each episode's ``stages``. ``lerobot-eval`` seeds the
episodes it counts (``--seed``, 1000 by default), so ``--seed=null`` gives no stage rates.
"""

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def stage_rates(episodes: list[dict[str, bool]]) -> dict[str, float]:
    """Percent of episodes that reached each stage, as ``pc_<stage>``; empty if no stage was logged."""
    names = list(dict.fromkeys(name for episode in episodes for name in episode))
    return {f"pc_{name}": float(np.mean([ep.get(name, False) for ep in episodes]) * 100) for name in names}


def read_stage_log(prefix: Path, n_episodes: int) -> list[dict[str, bool]]:
    """The stage flags of the first ``n_episodes`` seeded episodes logged under ``prefix``, by seed."""
    records = [
        json.loads(line)
        for path in sorted(prefix.parent.glob(f"{prefix.name}.*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    records.sort(key=lambda record: record["seed"])
    return [record["stages"] for record in records[:n_episodes]]


def eval_policy_all_with_stages(eval_policy_all: Callable[..., dict], envs: dict, **kwargs: Any) -> dict:
    """Run LeRobot's ``eval_policy_all`` and add the stage rates of every task, group and the overall run."""
    with tempfile.TemporaryDirectory(prefix="nexarm-stages-") as log_dir:
        prefixes = {}
        for group, tasks in envs.items():
            for task_id, vec_env in tasks.items():
                prefixes[group, task_id] = Path(log_dir) / f"{group}-{task_id}"
                # VectorEnv.call reaches each sub-env, also in AsyncVectorEnv worker processes.
                vec_env.call("log_stages_to", str(prefixes[group, task_id]))
        # eval_policy_all closes each env after its task, which writes any unfinished episode.
        info = eval_policy_all(envs=envs, **kwargs)
        stages = {key: read_stage_log(prefix, kwargs["n_episodes"]) for key, prefix in prefixes.items()}

    by_group: dict[str, list[dict[str, bool]]] = {}
    for task in info["per_task"]:
        episodes = stages.get((task["task_group"], task["task_id"]), [])
        if episodes:
            task["metrics"]["stages"] = episodes
            task["metrics"].update(stage_rates(episodes))
            by_group.setdefault(task["task_group"], []).extend(episodes)
    for group, episodes in by_group.items():
        info["per_group"][group].update(stage_rates(episodes))
    info["overall"].update(stage_rates([ep for episodes in by_group.values() for ep in episodes]))
    return info


def main() -> None:
    """``lerobot-eval`` with ``pc_<stage>`` rates in its log and ``eval_info.json``."""
    import functools

    from lerobot.scripts import lerobot_eval

    original = lerobot_eval.eval_policy_all
    # eval_main calls eval_policy_all by its module-level name.
    setattr(  # noqa: B010
        lerobot_eval, "eval_policy_all", functools.partial(eval_policy_all_with_stages, original)
    )
    lerobot_eval.main()


if __name__ == "__main__":
    main()
