# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

.PHONY: build-user test-end-to-end test-nexarm-ete

PYTHON_PATH := $(shell which python)

# If a virtual environment exists, use it
ifneq ($(wildcard .venv/bin/python),)
	PYTHON_PATH := $(abspath .venv/bin/python)
endif

export PATH := $(dir $(PYTHON_PATH)):$(PATH)

DEVICE ?= cpu

build-user:
	docker build -f docker/Dockerfile.user -t lerobot-user .

# End-to-end tests. Upstream's ACT/diffusion/TDMPC/SmolVLA train-and-eval targets live in huggingface/lerobot.
test-end-to-end:
	${MAKE} DEVICE=$(DEVICE) test-nexarm-ete

test-nexarm-ete:
	uv run pytest tests/envs/test_nexarm_sim_gym_integration.py -svv
