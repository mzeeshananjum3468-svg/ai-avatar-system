import fnmatch
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class AvatarProcessor:
    """Process and prepare avatar images for animation"""

    def __init__(self):
        self.resolution = settings.AVATAR_RESOLUTION

    def cleanup_temp_files(self, temp_dir: Optional[str | Path] = None) -> int:
        """Remove stale avatar temp artifacts from the system temp directory."""
        target_dir = Path(temp_dir or tempfile.gettempdir())
        if not target_dir.exists():
            return 0

        patterns = ("*_original.*", "*_processed.*", "*_thumb.*", "*.frame.jpg")
        cleaned = 0
        for file_path in target_dir.iterdir():
            if not file_path.is_file():
                continue
            if any(fnmatch.fnmatch(file_path.name, pattern) for pattern in patterns):
                file_path.unlink(missing_ok=True)
                cleaned += 1

        return cleaned

    def _is_video_input(self, input_path: str, source_type: Optional[str] = None) -> bool:
        if source_type == "video":
            return True
        if source_type == "image":
            return False

        suffix = Path(input_path).suffix.lower()
        return suffix in {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
            ".m4v",
            ".mpg",
            ".mpeg",
            ".wmv",
        }

    def _extract_first_frame(self, input_path: str, output_frame_path: Path) -> bool:
        capture = cv2.VideoCapture(input_path)
        try:
            success, frame = capture.read()
            if not success or frame is None:
                return False

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(output_frame_path, quality=95)
            return True
        finally:
            capture.release()

    async def process_media(
        self, input_path: str, output_path: str, source_type: Optional[str] = None
    ) -> Tuple[str, dict]:
        """Process an uploaded image or extract the first frame from a video."""
        if self._is_video_input(input_path, source_type):
            logger.info(f"Processing avatar video: {input_path}")
            temp_frame_path = Path(output_path).with_suffix(".frame.jpg")
            temp_frame_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._extract_first_frame(input_path, temp_frame_path):
                raise ValueError("Could not read a frame from the supplied video")
            try:
                output, metadata = await self.process_image(str(temp_frame_path), output_path)
                metadata["source_type"] = "video"
                metadata["source_path"] = input_path
                return output, metadata
            finally:
                temp_frame_path.unlink(missing_ok=True)

        output, metadata = await self.process_image(input_path, output_path)
        metadata["source_type"] = source_type or "image"
        metadata["source_path"] = input_path
        return output, metadata

    async def process_image(self, image_path: str, output_path: str) -> Tuple[str, dict]:
        """
        Process uploaded avatar image

        Args:
            image_path: Path to input image
            output_path: Path to save processed image

        Returns:
            Tuple of (output_path, metadata)
        """
        try:
            logger.info(f"Processing avatar image: {image_path}")

            # Load image
            image = Image.open(image_path)

            # Convert to RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Get original dimensions
            orig_width, orig_height = image.size

            # Detect face for metadata only — the full original picture is
            # kept intact (no crop) since it's fed as-is to both MuseTalk
            # (lip-sync) and LivePortrait (idle/thinking loops), and either
            # one cropping the source before those models see it would
            # discard parts of the picture the user actually uploaded.
            face_box = await self._detect_face(np.array(image))
            if face_box:
                logger.info(f"Face detected: {face_box}")
            else:
                logger.warning("No face detected in uploaded avatar image")

            # Downscale only if the image is larger than the configured cap,
            # preserving the original aspect ratio — never crop or pad.
            longest_side = max(orig_width, orig_height)
            if longest_side > self.resolution:
                scale = self.resolution / longest_side
                new_size = (round(orig_width * scale), round(orig_height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Enhance image
            image = await self._enhance_image(image)

            # Save processed image
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, quality=95)

            # Create thumbnail
            thumbnail_path = output_path.replace(".", "_thumb.")
            thumbnail = image.copy()
            thumbnail.thumbnail((256, 256), Image.Resampling.LANCZOS)
            thumbnail.save(thumbnail_path, quality=85)

            metadata = {
                "original_size": (orig_width, orig_height),
                "processed_size": image.size,
                "face_detected": face_box is not None,
                "thumbnail_path": thumbnail_path,
            }

            logger.info(f"Avatar processed successfully: {output_path}")
            return output_path, metadata

        except Exception as e:
            logger.error(f"Failed to process avatar image: {e}")
            raise

    async def _detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in image using OpenCV"""
        try:
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            if len(faces) > 0:
                # Return largest face
                return tuple(max(faces, key=lambda f: f[2] * f[3]))

            return None

        except Exception as e:
            logger.warning(f"Face detection error: {e}")
            return None

    async def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality"""
        try:
            from PIL import ImageEnhance

            # Slightly enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)

            # Slightly enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)

            return image

        except Exception as e:
            logger.warning(f"Image enhancement error: {e}")
            return image

    async def create_thumbnail(
        self, image_path: str, thumbnail_path: str, size: Tuple[int, int] = (256, 256)
    ) -> str:
        """Create thumbnail from image"""
        try:
            image = Image.open(image_path)
            image.thumbnail(size, Image.Resampling.LANCZOS)
            image.save(thumbnail_path, quality=85)
            return thumbnail_path

        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            raise


# Global instance
avatar_processor = AvatarProcessor()
