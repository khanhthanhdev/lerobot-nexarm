# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""jobs and models routes: training queue, metrics, checkpoints, model discovery."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from nexarm_service.dataset_store import DatasetNotFoundError
from nexarm_service.jobs import Job, MetricPoint, TrainRequest
from nexarm_service.models import ModelInfo
from nexarm_service.operations import Operation

router = APIRouter()


def _job_missing(job_id: str) -> HTTPException:
    return HTTPException(404, f"no job {job_id!r}")


@router.post("/jobs", status_code=201, tags=["jobs"])
def submit_job(body: TrainRequest, request: Request) -> Job:
    state = request.app.state
    if body.runner == "local":
        try:
            state.dataset_store.path(body.dataset_repo_id)
        except DatasetNotFoundError as exc:
            raise HTTPException(404, f"no local dataset {body.dataset_repo_id!r}") from exc
    try:
        return state.jobs.submit(body)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/jobs", tags=["jobs"])
def list_jobs(request: Request) -> list[Job]:
    return request.app.state.jobs.list()


@router.get("/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str, request: Request) -> Job:
    try:
        return request.app.state.jobs.get(job_id)
    except KeyError as exc:
        raise _job_missing(job_id) from exc


@router.delete("/jobs/{job_id}", tags=["jobs"])
def cancel_job(job_id: str, request: Request) -> Job:
    try:
        return request.app.state.jobs.cancel(job_id)
    except KeyError as exc:
        raise _job_missing(job_id) from exc


@router.get("/jobs/{job_id}/logs", tags=["jobs"])
def job_logs(job_id: str, request: Request, tail: int = 200, offset: int | None = None) -> list[str]:
    try:
        return request.app.state.jobs.logs(job_id, tail, offset)
    except KeyError as exc:
        raise _job_missing(job_id) from exc


@router.get("/jobs/{job_id}/metrics", tags=["jobs"])
def job_metrics(job_id: str, request: Request) -> list[MetricPoint]:
    try:
        return request.app.state.jobs.metrics(job_id)
    except KeyError as exc:
        raise _job_missing(job_id) from exc


@router.get("/jobs/{job_id}/checkpoints", tags=["jobs"])
def job_checkpoints(job_id: str, request: Request) -> list[dict[str, Any]]:
    try:
        return request.app.state.jobs.checkpoints(job_id)
    except KeyError as exc:
        raise _job_missing(job_id) from exc


class ModelImport(BaseModel):
    repo_id: str
    revision: str | None = None


@router.get("/models", tags=["models"])
def list_models(request: Request) -> list[ModelInfo]:
    return request.app.state.models.list_local()


@router.get("/models/hub", tags=["models"])
def list_hub_models(author: str, request: Request) -> list[ModelInfo]:
    try:
        return request.app.state.models.list_hub(author)
    except Exception as exc:  # network and auth failures surface as one gateway error
        raise HTTPException(502, f"could not list Hub models: {exc}") from exc


@router.post("/models/import", status_code=202, tags=["models"])
def import_model(body: ModelImport, request: Request) -> Operation:
    state = request.app.state
    return state.operations.submit(
        f"import model {body.repo_id}", lambda: state.models.import_from_hub(body.repo_id, body.revision)
    )


@router.get("/jobs/{job_id}/checkpoints/{step}/archive", tags=["jobs"], response_class=FileResponse)
def checkpoint_archive(job_id: str, step: str, request: Request) -> FileResponse:
    """A checkpoint's ``pretrained_model`` directory as a tar.gz (how remote training returns results)."""
    from nexarm_service.archive import pack

    state = request.app.state
    try:
        checkpoints = state.jobs.checkpoints(job_id)
    except KeyError as exc:
        raise _job_missing(job_id) from exc
    match = next(
        (c for c in checkpoints if str(c["step"]) == step or Path(c["path"]).parent.name == step), None
    )
    if match is None:
        raise HTTPException(404, f"job {job_id} has no checkpoint {step!r}")
    state.scratch.mkdir(parents=True, exist_ok=True)
    archive = pack(Path(match["path"]), state.scratch)
    return FileResponse(
        archive, media_type="application/gzip", background=BackgroundTask(archive.unlink, True)
    )
