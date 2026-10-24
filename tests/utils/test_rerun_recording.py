#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import socket

import pytest

rr = pytest.importorskip("rerun", reason="rerun-sdk is required (install lerobot[viz])")

from lerobot_robot_nexarm.rerun_recording import init_rerun  # noqa: E402


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _logged_file_size(path, n_scalars: int) -> int:
    for i in range(n_scalars):
        rr.set_time("step", sequence=i)
        rr.log("observation/joint", rr.Scalars(float(i)))
    rr.disconnect()
    return path.stat().st_size


def test_save_path_keeps_data_while_streaming_to_a_viewer(tmp_path):
    """The .rrd must receive the logged data, not only a header, when a viewer connection is also open."""
    header_only = tmp_path / "header.rrd"
    init_rerun("rerun_save_test", ip="127.0.0.1", port=_free_port(), save_path=header_only)
    header_size = _logged_file_size(header_only, 0)

    recording = tmp_path / "nested" / "session.rrd"
    init_rerun("rerun_save_test", ip="127.0.0.1", port=_free_port(), save_path=recording)
    recorded_size = _logged_file_size(recording, 200)

    assert recorded_size > header_size + 2000


def test_rollout_flag_is_removed_before_lerobot_rollout_parses_it():
    from lerobot_robot_nexarm.rerun_recording import split_save_path

    argv = ["--robot.type=nexarm_sim", "--display_data=true", "--rerun_save_path=out/run.rrd"]
    assert split_save_path(argv) == (["--robot.type=nexarm_sim", "--display_data=true"], "out/run.rrd")
    assert split_save_path(["--robot.type=nexarm_sim"]) == (["--robot.type=nexarm_sim"], None)
    with pytest.raises(ValueError, match="requires --display_data=true"):
        split_save_path(["--rerun_save_path=out/run.rrd"])
