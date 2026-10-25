# How to contribute to LeRobot NexArm

Everyone is welcome to contribute. Please respect our [code of conduct](./CODE_OF_CONDUCT.md) and our [AI policy](./AI_POLICY.md).

This repository holds the NexArm plugins, research policies, sim assets and tools; LeRobot itself is an upstream dependency. Bugs and features in LeRobot's own code (training, datasets, upstream robots and policies) belong in [huggingface/lerobot](https://github.com/huggingface/lerobot).

## Ways to Contribute

- **Fixing issues:** resolve bugs or improve the NexArm code.
- **New features:** NexArm hardware support, sim tasks, tools.
- **Research policies:** add a LeRobot policy plugin under `policies/`.
- **Documentation:** improve the NexArm guides and examples.
- **Feedback:** open an issue for bugs or wanted features.

## Development Setup

### 1. Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-handle>/lerobot-nexarm.git
cd lerobot-nexarm
git remote add upstream https://github.com/Hiwonder-official/lerobot-nexarm.git
```

### 2. Environment Installation

Follow [Dependency profiles in the install guide](./docs/getting-started/install.md#dependency-profiles). All user guides are indexed in [`docs/README.md`](./docs/README.md). `uv sync` installs the pinned `lerobot`, the NexArm plugins and the B-spline policy plugin; `--extra turbovla` adds TurboVLA and its plugin. Always run commands through `uv run`.

## Running Tests & Quality Checks

### Code Style (Pre-commit)

```bash
uv run pre-commit install --install-hooks             # once per clone
uv run pre-commit run --all-files --show-diff-on-failure
```

### Running Tests

Pull the test artifacts with **git-lfs** first:

```bash
git lfs install
git lfs pull
```

Run the suite (hardware is mocked; the MuJoCo rendering tests need OpenGL/EGL):

```bash
uv run pytest tests policies -sv
uv run pytest tests/robots/test_nexarm_follower.py -sv   # one file
make test-nexarm-ete                                    # NexArm sim end-to-end
```

## Upgrading LeRobot

`lerobot` is pinned to an upstream commit in `[tool.uv.sources]` of the root `pyproject.toml`. To move to a newer one:

1. Set `rev` to the new upstream commit (prefer a release tag's commit).
2. Run `uv lock`, then `uv sync --locked` with the extras you use.
3. Run `uv run pytest tests policies`, `make test-nexarm-ete` and `uv run pre-commit run --all-files`.
4. Check the CLIs still see the plugins: `uv run lerobot-record --help` lists `nexarm_follower`, `uv run lerobot-eval --env.type=nexarm --help` parses, and `lerobot-nexarm-eval`, `lerobot-nexarm-rollout` and `lerobot-nexarm-service --dump-openapi` start.
5. Where a test fails because LeRobot changed an API, fix the plugin. The wrappers `lerobot-nexarm-eval` and `lerobot-nexarm-rollout` replace a function inside `lerobot.scripts.lerobot_eval` / `lerobot_rollout`, so check those names first.

mypy does not see LeRobot's types (no `py.typed`), so the tests are what catch API changes.

## Submitting Issues & Pull Requests

- **Issues:** use the [issue tracker](https://github.com/Hiwonder-official/lerobot-nexarm/issues/new).
- **Pull requests:** rebase on `upstream/main`, use a descriptive branch (don't work on `main`), run `pre-commit` and the tests locally, and follow the [PR template](https://github.com/Hiwonder-official/lerobot-nexarm/blob/main/.github/PULL_REQUEST_TEMPLATE.md).

Thank you for contributing!
