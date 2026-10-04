"""Regression coverage for imports of unrelated optional policies."""

import subprocess
import sys
import textwrap


def test_act_import_does_not_load_groot_processors_without_opencv():
    # Use a fresh interpreter because other tests may already have imported cv2
    # or registered GR00T processor steps in this process.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import importlib.abc
                import sys

                class BlockOpenCV(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "cv2" or fullname.startswith("cv2."):
                            raise ModuleNotFoundError("No module named 'cv2'", name=fullname)

                sys.meta_path.insert(0, BlockOpenCV())
                from lerobot.policies.act import ACTConfig
                from lerobot.policies.groot import GrootConfig
                from lerobot.policies import make_policy_config

                assert isinstance(make_policy_config("act"), ACTConfig)
                assert isinstance(make_policy_config("groot"), GrootConfig)
                assert "lerobot.policies.groot.processor_groot" not in sys.modules
                assert "lerobot.policies.groot.modeling_groot" not in sys.modules
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
