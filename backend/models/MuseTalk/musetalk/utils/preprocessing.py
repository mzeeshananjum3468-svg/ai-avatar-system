# preprocessing.py — rewritten for Python 3.12 / CPU-only compatibility
# Uses face_alignment (pip install face-alignment) instead of mmpose/mmcv.
import os
import json
import pickle
import numpy as np
import cv2
import torch
from tqdm import tqdm
import face_alignment

# Initialise face alignment on CPU or GPU automatically
_device = "cuda" if torch.cuda.is_available() else "cpu"
_fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device=_device,
)

# Sentinel value used by callers when no face is detected in a frame
coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def resize_landmark(landmark, w, h, new_w, new_h):
    landmark_norm = landmark / np.array([w, h])
    return landmark_norm * np.array([new_w, new_h])


def read_imgs(img_list):
    frames = []
    print("reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def _detect_face_bbox(frame_bgr, bbox_shift=0):
    """Detect face bounding box using 68-point landmarks. Returns (x1,y1,x2,y2) or None."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preds = _fa.get_landmarks(frame_rgb)
    if preds is None or len(preds) == 0:
        return None
    lm = preds[0]  # (68, 2)
    x1, y1 = int(lm[:, 0].min()), int(lm[:, 1].min())
    x2, y2 = int(lm[:, 0].max()), int(lm[:, 1].max())
    y1 = max(0, y1 + bbox_shift)
    if x2 <= x1 or y2 <= y1 or x1 < 0:
        return None
    return (x1, y1, x2, y2)


def get_landmark_and_bbox(img_list, upperbondrange=0):
    """
    Detect face bounding boxes for all images.
    Returns (coords_list, frames).
    """
    frames = read_imgs(img_list)
    coords_list = []
    print(f"Getting face bounding boxes (bbox_shift={upperbondrange})..." if upperbondrange != 0
          else "Getting face bounding boxes...")
    for frame in tqdm(frames):
        bbox = _detect_face_bbox(frame, bbox_shift=upperbondrange)
        coords_list.append(bbox if bbox is not None else coord_placeholder)
    return coords_list, frames


def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    deltas = []
    for frame in tqdm(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = _fa.get_landmarks(frame_rgb)
        if preds is None or len(preds) == 0:
            continue
        lm = preds[0]
        deltas.append(int(lm[:, 1].max()) - int(lm[:, 1].mean()))
    if not deltas:
        return "No faces detected"
    avg = int(sum(deltas) / len(deltas))
    return f"Total frame: {len(frames)}  Adjust range: [-{avg}~{avg}]  current value: {upperbondrange}"
