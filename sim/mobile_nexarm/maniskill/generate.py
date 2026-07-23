#!/usr/bin/env python
"""Launch registered ManiSkill environments and retain only successful episodes."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def require_maniskill() -> None:
    if importlib.util.find_spec("mani_skill") is None:
        raise RuntimeError(
            "ManiSkill is unavailable. Run `uv sync` in sim/mobile_nexarm/maniskill "
            "and invoke this script with that environment."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    require_maniskill()
    from agent import register_agent

    agent = register_agent()
    print(f"registered ManiSkill agent {agent.uid} from {agent.urdf_path}")
    if args.preflight:
        return
    raise RuntimeError(
        "The ManiSkill SDK is available, but no GPU episode adapter was selected. "
        "Instantiate the registered agent in a project-specific BaseEnv and pass it "
        "to PickPlaceSkill; neutral artifact writing is backend-independent."
    )


if __name__ == "__main__":
    main()
