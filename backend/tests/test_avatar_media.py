import asyncio
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.animator import avatar_animator
from app.services.avatar_processor import avatar_processor


def test_cleanup_temp_files_removes_avatar_artifacts(tmp_path: Path):
    stale_original = tmp_path / "demo_avatar_original.jpg"
    stale_processed = tmp_path / "demo_avatar_processed.jpg"
    keep_file = tmp_path / "keep.txt"

    stale_original.write_bytes(b"old")
    stale_processed.write_bytes(b"old")
    keep_file.write_bytes(b"keep")

    cleaned_count = avatar_processor.cleanup_temp_files(tmp_path)

    assert cleaned_count == 2
    assert not stale_original.exists()
    assert not stale_processed.exists()
    assert keep_file.exists()


def test_process_media_accepts_video_input(tmp_path: Path):
    video_path = tmp_path / "avatar.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (32, 32),
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    writer.write(frame)
    writer.release()

    output_path = tmp_path / "processed.jpg"
    output, metadata = asyncio.run(avatar_processor.process_media(str(video_path), str(output_path)))

    assert output == str(output_path)
    assert output_path.exists()
    assert metadata["source_type"] == "video"


def test_prepare_avatar_source_extracts_frame_from_video(tmp_path: Path):
    video_path = tmp_path / "avatar.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (32, 32),
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    writer.write(frame)
    writer.release()

    prepared_path = avatar_animator._prepare_avatar_source(str(video_path))

    assert Path(prepared_path).exists()
    assert Path(prepared_path).suffix == ".jpg"


@pytest.mark.asyncio
async def test_upload_avatar_stores_original_video_and_metadata(client, test_engine, monkeypatch):
    from app.api.v1 import avatars
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    uploaded_keys = []

    async def fake_process_media(input_path, output_path, source_type=None):
        Path(output_path).write_bytes(b"processed-image")
        return output_path, {"thumbnail_path": ""}

    async def fake_upload_file(file_data, key, content_type="application/octet-stream", metadata=None):
        uploaded_keys.append((key, file_data, content_type))
        return f"http://test-storage/{key}"

    monkeypatch.setattr(avatars.avatar_processor, "process_media", fake_process_media)
    monkeypatch.setattr(avatars.storage_service, "upload_file", fake_upload_file)
    # storage_service.download_file isn't stubbed, so the background
    # idle/thinking job fails fast (no such local file) and marks the avatar
    # "failed" — that write goes through AsyncSessionLocal, which by default
    # points at the real configured DB rather than the test's in-memory one,
    # so point it at the test engine (same pattern as test_ws_auth.py).
    sm = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(avatars, "AsyncSessionLocal", sm)

    response = await client.post(
        "/api/v1/avatars/upload",
        files={"file": ("avatar.mp4", b"video-bytes", "video/mp4")},
        data={"name": "Video Avatar"},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    # The response is serialized right after the row is created — before the
    # background LivePortrait job runs — so it's still "processing" here.
    assert body["status"] == "processing"
    assert body["avatar_metadata"]["source_type"] == "video"
    assert any(key.startswith("avatars/") and key.endswith("/source.mp4") for key, _, _ in uploaded_keys)
    assert any(key.startswith("avatars/") and key.endswith("/image.jpg") for key, _, _ in uploaded_keys)
