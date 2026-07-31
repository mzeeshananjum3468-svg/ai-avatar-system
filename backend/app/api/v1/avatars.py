import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.users import get_current_user
from app.database import AsyncSessionLocal, get_db
from app.models import Avatar, User
from app.schemas import AvatarMetadataUpdate, AvatarRename, AvatarResponse
from app.services.avatar_processor import avatar_processor
from app.services.liveportrait import liveportrait_service
from app.services.storage import storage_service

logger = logging.getLogger(__name__)
router = APIRouter()
TMPDIR = Path(tempfile.gettempdir())
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv")


def _user_id(current_user: Optional[User]) -> str:
    return current_user.id if current_user else "demo-user"


def _validate_uuid(avatar_id: str) -> None:
    """
    Reject anything that isn't a UUID to keep S3 keys + filesystem paths safe.

    Returns 404 (not 400) for malformed IDs: a non-UUID can never name an
    existing avatar, so "not found" is the correct REST semantics and it
    avoids leaking the fact that IDs are UUIDs to a probing client.
    """
    try:
        uuid.UUID(avatar_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="Avatar not found")


def _detect_media_type(content_type: Optional[str], filename: Optional[str]) -> str:
    content_type = (content_type or "").lower()
    filename = (filename or "").lower()

    if content_type.startswith("image/") or any(filename.endswith(ext) for ext in IMAGE_EXTENSIONS):
        return "image"
    if content_type.startswith("video/") or any(filename.endswith(ext) for ext in VIDEO_EXTENSIONS):
        return "video"

    raise HTTPException(
        status_code=400,
        detail="File must be an image or video (JPG, PNG, WEBP, MP4, MOV, AVI, WEBM)",
    )


async def _set_avatar_fields(avatar_id: str, **fields) -> None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
        avatar = result.scalar_one_or_none()
        if avatar is None:
            # Avatar was deleted while generation was in flight.
            return
        for key, value in fields.items():
            setattr(avatar, key, value)
        await db.commit()


async def _generate_idle_thinking_videos(avatar_id: str, image_key: str) -> None:
    """Background job: run LivePortrait to build the idle/thinking loop
    videos for a freshly-uploaded avatar, then flip it to "ready" once both
    are attached. The avatar is created with status="processing" and
    sessions.create_session refuses to start a chat until it reaches
    "ready", so a half-built avatar (missing either loop video) can never be
    used.

    Scheduled to run after the upload response has already gone out, so a
    failure here (LivePortrait not installed, GPU OOM, etc.) never surfaces
    to the uploading request — the avatar is instead marked "failed".
    """
    if not liveportrait_service.available:
        logger.warning(f"LivePortrait unavailable — marking avatar {avatar_id} failed")
        await _set_avatar_fields(avatar_id, status="failed")
        return

    temp_image = TMPDIR / f"{avatar_id}_liveportrait_source.jpg"
    idle_path: Optional[str] = None
    idle_reversed_path: Optional[str] = None
    thinking_path: Optional[str] = None
    thinking_reversed_path: Optional[str] = None
    try:
        temp_image.write_bytes(await storage_service.download_file(image_key))

        (
            idle_path,
            idle_reversed_path,
            thinking_path,
            thinking_reversed_path,
        ) = await liveportrait_service.generate_idle_and_thinking(str(temp_image), avatar_id)
        if not idle_path or not thinking_path:
            await _set_avatar_fields(avatar_id, status="failed")
            return

        idle_url = await storage_service.upload_file(
            Path(idle_path).read_bytes(), f"avatars/{avatar_id}/idle.mp4", content_type="video/mp4"
        )
        thinking_url = await storage_service.upload_file(
            Path(thinking_path).read_bytes(),
            f"avatars/{avatar_id}/thinking.mp4",
            content_type="video/mp4",
        )
        # Best-effort — a missing reversed clip just means the client falls
        # back to a plain forward loop for that clip, not a failed avatar.
        idle_reversed_url = None
        if idle_reversed_path:
            idle_reversed_url = await storage_service.upload_file(
                Path(idle_reversed_path).read_bytes(),
                f"avatars/{avatar_id}/idle_reversed.mp4",
                content_type="video/mp4",
            )
        thinking_reversed_url = None
        if thinking_reversed_path:
            thinking_reversed_url = await storage_service.upload_file(
                Path(thinking_reversed_path).read_bytes(),
                f"avatars/{avatar_id}/thinking_reversed.mp4",
                content_type="video/mp4",
            )

        await _set_avatar_fields(
            avatar_id,
            idle_video_url=idle_url,
            thinking_video_url=thinking_url,
            idle_video_reversed_url=idle_reversed_url,
            thinking_video_reversed_url=thinking_reversed_url,
            status="ready",
        )

        logger.info(f"Idle/thinking videos ready for avatar {avatar_id}")
    except Exception:
        logger.exception(f"Failed to generate idle/thinking videos for avatar {avatar_id}")
        await _set_avatar_fields(avatar_id, status="failed")
    finally:
        temp_image.unlink(missing_ok=True)
        if idle_path:
            Path(idle_path).unlink(missing_ok=True)
        if idle_reversed_path:
            Path(idle_reversed_path).unlink(missing_ok=True)
        if thinking_path:
            Path(thinking_path).unlink(missing_ok=True)
        if thinking_reversed_path:
            Path(thinking_reversed_path).unlink(missing_ok=True)


@router.post("/upload", response_model=AvatarResponse, status_code=status.HTTP_201_CREATED)
async def upload_avatar(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Upload and process an avatar image or video."""
    source_type = _detect_media_type(file.content_type, file.filename)

    file_data: bytes = await file.read()  # type: ignore[assignment]
    if len(file_data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File must be under 10 MB")

    avatar_id = str(uuid.uuid4())
    suffix = Path(file.filename or ("avatar.mp4" if source_type == "video" else "avatar.jpg")).suffix.lower()
    if not suffix:
        suffix = ".mp4" if source_type == "video" else ".jpg"
    temp_orig = TMPDIR / f"{avatar_id}_original{suffix}"
    temp_processed = TMPDIR / f"{avatar_id}_processed.jpg"
    metadata: dict = {}
    source_key = f"avatars/{avatar_id}/source{suffix}"
    source_content_type = file.content_type or ("video/mp4" if source_type == "video" else "image/jpeg")

    try:
        avatar_processor.cleanup_temp_files(TMPDIR)
        temp_orig.write_bytes(file_data)

        source_url = await storage_service.upload_file(
            file_data,
            source_key,
            content_type=source_content_type,
        )

        _, metadata = await avatar_processor.process_media(
            str(temp_orig), str(temp_processed), source_type=source_type
        )

        metadata["source_type"] = source_type
        metadata["source_url"] = source_url
        metadata["source_key"] = source_key

        image_key = f"avatars/{avatar_id}/image.jpg"
        image_url = await storage_service.upload_file(
            temp_processed.read_bytes(), image_key, content_type="image/jpeg"
        )

        # Resolve the thumbnail path defensively: an empty/missing value would
        # make Path("") == Path(".") (the cwd), so guard against it explicitly
        # and fall back to the processed image when there's no real thumbnail.
        thumb_value = metadata.get("thumbnail_path") or ""
        thumb_file = Path(thumb_value) if thumb_value else None
        thumb_key = f"avatars/{avatar_id}/thumbnail.jpg"
        thumb_bytes = (
            thumb_file.read_bytes()
            if thumb_file and thumb_file.is_file()
            else temp_processed.read_bytes()
        )
        thumbnail_url = await storage_service.upload_file(
            thumb_bytes, thumb_key, content_type="image/jpeg"
        )

    except HTTPException:
        raise
    except Exception as e:
        from PIL import UnidentifiedImageError

        if isinstance(e, UnidentifiedImageError):
            # Client sent something that isn't a decodable image — their
            # fault, not ours.
            raise HTTPException(
                status_code=400, detail="File is not a valid image (JPG, PNG, WEBP)"
            )
        logger.error(f"Avatar processing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process avatar")
    finally:
        temp_orig.unlink(missing_ok=True)
        temp_processed.unlink(missing_ok=True)
        # is_file() guards against the Path("") == "." footgun — never unlink
        # a directory.
        thumb_value = metadata.get("thumbnail_path") or ""
        if thumb_value:
            thumb_file = Path(thumb_value)
            if thumb_file.is_file():
                thumb_file.unlink(missing_ok=True)

    avatar_storage_key = source_key if source_type == "video" else image_key
    avatar_storage_url = source_url if source_type == "video" else image_url

    avatar = Avatar(
        id=avatar_id,
        user_id=_user_id(current_user),
        name=name,
        image_url=avatar_storage_url,
        thumbnail_url=thumbnail_url,
        s3_key=avatar_storage_key,
        status="processing",
        avatar_metadata=metadata,
    )
    db.add(avatar)
    await db.commit()
    await db.refresh(avatar)

    # Kick off idle/thinking loop video generation in the background — the
    # image is already in storage, so this only needs its key, not the
    # (already-deleted) temp file. The avatar stays "processing" (unusable
    # for chat, see sessions.create_session) until both videos land, and
    # flips to "failed" if LivePortrait isn't configured or generation
    # errors out.
    background_tasks.add_task(_generate_idle_thinking_videos, avatar_id, image_key)

    logger.info(f"Avatar created: {avatar_id} for user {_user_id(current_user)}")
    return avatar


@router.get("/", response_model=List[AvatarResponse])
async def list_avatars(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List avatars belonging to the current user."""
    uid = _user_id(current_user)
    result = await db.execute(
        select(Avatar)
        .where(Avatar.user_id == uid)
        .offset(skip)
        .limit(limit)
        .order_by(Avatar.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{avatar_id}", response_model=AvatarResponse)
async def get_avatar(
    avatar_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to access this avatar")
    return avatar


@router.put("/{avatar_id}/voice", response_model=AvatarResponse)
async def set_avatar_voice(
    avatar_id: str,
    voice_id: Optional[str] = Query(
        default=None,
        description="Voice profile ID to assign. Omit or pass an empty string to unassign.",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Assign (or clear) a voice profile on an avatar."""
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this avatar")

    try:
        normalized = (voice_id or "").strip() or None
        avatar.voice_id = normalized
        await db.commit()
        await db.refresh(avatar)
        logger.info(f"Avatar {avatar_id} voice set to: {normalized!r}")
        return avatar
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to set voice for avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update avatar voice")


@router.patch("/{avatar_id}/metadata", response_model=AvatarResponse)
async def update_avatar_metadata(
    avatar_id: str,
    payload: AvatarMetadataUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Merge an allowlist of metadata fields into avatar_metadata."""
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this avatar")

    existing: dict = avatar.avatar_metadata or {}
    if isinstance(existing, str):
        import json as _json

        try:
            existing = _json.loads(existing)
        except Exception:
            existing = {}

    # Only merge fields the caller actually set (exclude_unset=True keeps the
    # PATCH semantics — omitted fields are left untouched, not nulled out).
    update_data = payload.model_dump(exclude_unset=True)
    try:
        # Assign a NEW dict — mutating the ORM-held dict in place and
        # re-assigning the same object defeats SQLAlchemy's change detection
        # (old is new → no UPDATE emitted), so the metadata silently never
        # persisted even though the endpoint returned 200.
        avatar.avatar_metadata = {**existing, **update_data}
        await db.commit()
        await db.refresh(avatar)
        logger.info(f"Avatar {avatar_id} metadata updated: {list(update_data.keys())}")
        return avatar
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to update metadata for avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update avatar metadata")


@router.patch("/{avatar_id}/name", response_model=AvatarResponse)
async def rename_avatar(
    avatar_id: str,
    payload: AvatarRename,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Rename an avatar."""
    _validate_uuid(avatar_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this avatar")
    try:
        avatar.name = name
        await db.commit()
        await db.refresh(avatar)
        logger.info(f"Avatar {avatar_id} renamed to: {name!r}")
        return avatar
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to rename avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to rename avatar")


@router.delete("/{avatar_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_avatar(
    avatar_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to delete this avatar")

    # Delete the DB row first (sessions/messages/conversations cascade), THEN
    # the stored files — if the DB delete fails we haven't orphaned the row by
    # removing its image out from under it.
    try:
        await db.delete(avatar)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to delete avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete avatar")

    try:
        await storage_service.delete_file(avatar.s3_key)
        await storage_service.delete_file(avatar.s3_key.replace("image.jpg", "thumbnail.jpg"))
        # MuseTalk's per-avatar face/latent cache — see animator.py
        # _animate_musetalk, which names it "<source-file>.musetalk_cache.pkl".
        await storage_service.delete_file(f"{avatar.s3_key}.musetalk_cache.pkl")
        if avatar.idle_video_url:
            await storage_service.delete_file(f"avatars/{avatar_id}/idle.mp4")
            await storage_service.delete_file(f"avatars/{avatar_id}/idle.mp4.musetalk_cache.pkl")
        if avatar.idle_video_reversed_url:
            await storage_service.delete_file(f"avatars/{avatar_id}/idle_reversed.mp4")
        if avatar.thinking_video_url:
            await storage_service.delete_file(f"avatars/{avatar_id}/thinking.mp4")
        if avatar.thinking_video_reversed_url:
            await storage_service.delete_file(f"avatars/{avatar_id}/thinking_reversed.mp4")
        metadata = avatar.avatar_metadata or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        source_key = metadata.get("source_key")
        if source_key:
            await storage_service.delete_file(f"{source_key}.musetalk_cache.pkl")
    except Exception as e:
        # Row is gone; leftover files are harmless and reaped by the cleanup task.
        logger.warning(f"Could not delete stored files for avatar {avatar_id}: {e}")

    logger.info(f"Avatar deleted: {avatar_id}")
