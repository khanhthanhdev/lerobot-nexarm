# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""datasets and operations routes: view, curate, edit, merge, Hub transfer."""

import shutil
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from lerobot.nexarm_service.dataset_ops import list_hub_datasets
from lerobot.nexarm_service.dataset_store import (
    DatasetNotFoundError,
    DatasetSummary,
    EpisodeSummary,
    JointsData,
    valid_repo_id,
)
from lerobot.nexarm_service.operations import Operation

router = APIRouter()


def _not_found(exc: Exception) -> HTTPException:
    return HTTPException(404, f"not found: {exc.args[0]}")


class EpisodesBody(BaseModel):
    episodes: list[int]


class DeleteEpisodes(EpisodesBody):
    output_repo_id: str


class SplitBody(BaseModel):
    splits: dict[str, float | list[int]]


class MergeBody(BaseModel):
    repo_ids: list[str]
    output_repo_id: str


class FeatureSpec(BaseModel):
    dtype: str
    shape: list[int]
    names: list[str] | None = None
    values: list[Any]


class ModifyFeatures(BaseModel):
    output_repo_id: str
    remove: list[str] = []
    add: dict[str, FeatureSpec] = {}


class NewName(BaseModel):
    output_repo_id: str


class UploadBody(BaseModel):
    private: bool = True


class ImportBody(BaseModel):
    repo_id: str
    revision: str | None = None


# -- listing and import ---------------------------------------------------------------------
@router.get("/datasets", tags=["datasets"])
def list_datasets(request: Request) -> list[DatasetSummary]:
    return request.app.state.dataset_store.list_local()


@router.get("/datasets/hub", tags=["datasets"])
def list_hub(author: str) -> list[dict[str, Any]]:
    try:
        return list_hub_datasets(author)
    except Exception as exc:  # network and auth failures surface as one gateway error
        raise HTTPException(502, f"could not list Hub datasets: {exc}") from exc


@router.post("/datasets/import", status_code=202, tags=["datasets"])
def import_dataset(body: ImportBody, request: Request) -> Operation:
    state = request.app.state
    return state.operations.submit(
        f"import {body.repo_id}", lambda: state.dataset_ops.import_from_hub(body.repo_id, body.revision)
    )


@router.post("/datasets/merge", status_code=202, tags=["datasets"])
def merge(body: MergeBody, request: Request) -> Operation:
    state = request.app.state
    return state.operations.submit(
        f"merge into {body.output_repo_id}",
        lambda: state.dataset_ops.merge(body.repo_ids, body.output_repo_id),
    )


# -- one dataset ----------------------------------------------------------------------------
@router.get("/datasets/{owner}/{name}", tags=["datasets"])
def dataset_info(owner: str, name: str, request: Request) -> dict[str, Any]:
    store = request.app.state.dataset_store
    try:
        return {"summary": store.summary(f"{owner}/{name}"), "info": store.info(f"{owner}/{name}")}
    except DatasetNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/datasets/{owner}/{name}/episodes", tags=["datasets"])
def episodes(owner: str, name: str, request: Request) -> list[EpisodeSummary]:
    try:
        return request.app.state.dataset_store.episodes(f"{owner}/{name}")
    except DatasetNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/datasets/{owner}/{name}/episodes/{episode}/joints", tags=["datasets"])
def joints(owner: str, name: str, episode: int, request: Request) -> JointsData:
    try:
        return request.app.state.dataset_store.joints(f"{owner}/{name}", episode)
    except DatasetNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get(
    "/datasets/{owner}/{name}/episodes/{episode}/frames/{camera}/{frame}",
    tags=["datasets"],
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
def frame(owner: str, name: str, episode: int, camera: str, frame: int, request: Request, width: int = 640):
    try:
        jpeg = request.app.state.dataset_store.frame_jpeg(f"{owner}/{name}", episode, camera, frame, width)
    except DatasetNotFoundError as exc:
        raise _not_found(exc) from exc
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/datasets/{owner}/{name}/excluded", tags=["datasets"])
def get_excluded(owner: str, name: str, request: Request) -> list[int]:
    try:
        return request.app.state.dataset_store.excluded(f"{owner}/{name}")
    except DatasetNotFoundError as exc:
        raise _not_found(exc) from exc


@router.put("/datasets/{owner}/{name}/excluded", tags=["datasets"])
def set_excluded(owner: str, name: str, body: EpisodesBody, request: Request) -> list[int]:
    try:
        return request.app.state.dataset_store.set_excluded(f"{owner}/{name}", body.episodes)
    except DatasetNotFoundError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


# -- edits (each writes a new dataset) --------------------------------------------------------
def _submit(request: Request, label: str, fn) -> Operation:
    return request.app.state.operations.submit(label, fn)


@router.post("/datasets/{owner}/{name}/delete-episodes", status_code=202, tags=["datasets"])
def delete_episodes(owner: str, name: str, body: DeleteEpisodes, request: Request) -> Operation:
    ops, repo = request.app.state.dataset_ops, f"{owner}/{name}"
    return _submit(
        request,
        f"delete episodes of {repo}",
        lambda: ops.delete_episodes(repo, body.episodes, body.output_repo_id),
    )


@router.post("/datasets/{owner}/{name}/drop-excluded", status_code=202, tags=["datasets"])
def drop_excluded(owner: str, name: str, body: NewName, request: Request) -> Operation:
    ops, repo = request.app.state.dataset_ops, f"{owner}/{name}"
    return _submit(
        request, f"drop excluded episodes of {repo}", lambda: ops.drop_excluded(repo, body.output_repo_id)
    )


@router.post("/datasets/{owner}/{name}/split", status_code=202, tags=["datasets"])
def split(owner: str, name: str, body: SplitBody, request: Request) -> Operation:
    ops, repo = request.app.state.dataset_ops, f"{owner}/{name}"
    return _submit(request, f"split {repo}", lambda: ops.split(repo, body.splits))


@router.post("/datasets/{owner}/{name}/modify-features", status_code=202, tags=["datasets"])
def modify_features(owner: str, name: str, body: ModifyFeatures, request: Request) -> Operation:
    ops, repo = request.app.state.dataset_ops, f"{owner}/{name}"
    add = {k: v.model_dump() for k, v in body.add.items()}
    return _submit(
        request,
        f"modify features of {repo}",
        lambda: ops.modify_features(repo, body.output_repo_id, body.remove, add),
    )


@router.post("/datasets/{owner}/{name}/rename", tags=["datasets"])
def rename(owner: str, name: str, body: NewName, request: Request) -> dict[str, Any]:
    from lerobot.nexarm_service.dataset_ops import DatasetEditError

    try:
        return request.app.state.dataset_ops.rename(f"{owner}/{name}", body.output_repo_id)
    except DatasetNotFoundError as exc:
        raise _not_found(exc) from exc
    except DatasetEditError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/datasets/{owner}/{name}/upload", status_code=202, tags=["datasets"])
def upload(owner: str, name: str, body: UploadBody, request: Request) -> Operation:
    ops, repo = request.app.state.dataset_ops, f"{owner}/{name}"
    return _submit(request, f"upload {repo}", lambda: ops.upload(repo, body.private))


# -- operations -----------------------------------------------------------------------------
@router.get("/operations", tags=["datasets"])
def list_operations(request: Request) -> list[Operation]:
    return request.app.state.operations.list()


@router.get("/operations/{op_id}", tags=["datasets"])
def get_operation(op_id: str, request: Request) -> Operation:
    try:
        return request.app.state.operations.get(op_id)
    except KeyError as exc:
        raise HTTPException(404, f"no operation {op_id!r}") from exc


# -- archives (remote training moves datasets between engines) ---------------------------------
@router.put("/datasets/{owner}/{name}/archive", status_code=201, tags=["datasets"])
async def upload_archive(owner: str, name: str, request: Request) -> DatasetSummary:
    """Receive a dataset as a tar.gz body. Refuses a name that already exists; never overwrites."""
    from starlette.concurrency import run_in_threadpool

    from lerobot.nexarm_service.archive import ArchiveError, extract

    state = request.app.state
    repo = f"{owner}/{name}"
    if not valid_repo_id(repo):
        raise HTTPException(422, f"{repo!r} is not a valid repo id")
    target = state.dataset_store.root / repo
    if target.exists():
        raise HTTPException(409, f"{repo} already exists on this engine")
    state.scratch.mkdir(parents=True, exist_ok=True)
    received = state.scratch / f"upload-{uuid.uuid4().hex}.tar.gz"
    try:
        with open(received, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        try:
            await run_in_threadpool(extract, received, target)
            summary = state.dataset_store.summary(repo)
        except (ArchiveError, DatasetNotFoundError, KeyError) as exc:
            shutil.rmtree(target, ignore_errors=True)
            raise HTTPException(422, f"rejected the dataset archive: {exc}") from exc
    finally:
        received.unlink(missing_ok=True)
    return summary


@router.get("/datasets/{owner}/{name}/archive", tags=["datasets"], response_class=FileResponse)
def download_archive(owner: str, name: str, request: Request) -> FileResponse:
    from lerobot.nexarm_service.archive import pack

    state = request.app.state
    try:
        path = state.dataset_store.path(f"{owner}/{name}")
    except DatasetNotFoundError as exc:
        raise _not_found(exc) from exc
    archive = pack(path, state.scratch)
    return FileResponse(
        archive, media_type="application/gzip", background=BackgroundTask(archive.unlink, True)
    )
