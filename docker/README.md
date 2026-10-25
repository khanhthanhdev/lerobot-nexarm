# Docker

`Dockerfile.user` builds a CPU image with this repository, the pinned upstream LeRobot and every extra
(`uv sync --locked --extra all`), plus the NexArm plugins and research policies from the default dependency
groups.

```bash
make build-user                    # docker build -f docker/Dockerfile.user -t lerobot-user .
docker run -it --rm lerobot-user
```

Run it from the repository root with the Git LFS files pulled (`git lfs pull`); the build copies the whole
checkout. GPU, ROCm, Jetson and benchmark images are maintained upstream in
[huggingface/lerobot](https://github.com/huggingface/lerobot/tree/main/docker).
