"""Offline vision engine for Local QC Inspector.

The engine intentionally uses only local OpenCV/Numpy processing.  Each profile
has its own reference image, layout, labelled images, and small statistical
model stored under qc_data/profiles.  No image leaves the computer.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_profile_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:60] or "inspection_profile"


def read_image_bytes(data: bytes) -> np.ndarray:
    """Read an upload and honour its phone-camera EXIF orientation."""
    with Image.open(io.BytesIO(data)) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        rgb = np.asarray(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def image_to_png_bytes(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Could not encode the annotated inspection image.")
    return encoded.tobytes()


def profile_dir(data_root: Path, profile_name: str) -> Path:
    return data_root / "profiles" / safe_profile_name(profile_name)


def config_path(data_root: Path, profile_name: str) -> Path:
    return profile_dir(data_root, profile_name) / "profile.json"


def list_profiles(data_root: Path) -> list[str]:
    root = data_root / "profiles"
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "profile.json").exists()
    )


def load_profile(data_root: Path, profile_name: str) -> dict[str, Any]:
    path = config_path(data_root, profile_name)
    if not path.exists():
        raise FileNotFoundError(f"Inspection profile '{profile_name}' does not exist.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_profile(data_root: Path, profile_name: str, profile: dict[str, Any]) -> None:
    destination = config_path(data_root, profile_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
    temporary.replace(destination)


def reset_profile(data_root: Path, profile_name: str) -> None:
    """Delete only the named profile; used by the UI after an explicit action."""
    target = profile_dir(data_root, profile_name)
    if target.exists():
        shutil.rmtree(target)


def default_items() -> list[dict[str, Any]]:
    """Default locations for the supplied seat/blower assembly.

    The numbers are normalized to the reference image.  Screw IDs run clockwise
    from the upper-left screw; clip IDs run left-to-right.  The Layout tab lets
    an operator fine-tune every rectangle without editing code.
    """
    screw_centers = [
        (0.356, 0.129),
        (0.402, 0.109),
        (0.553, 0.110),
        (0.593, 0.137),
        (0.594, 0.236),
        (0.552, 0.254),
        (0.381, 0.254),
        (0.356, 0.236),
    ]
    items: list[dict[str, Any]] = []
    for index, (center_x, center_y) in enumerate(screw_centers, start=1):
        items.append(
            {
                "id": f"Screw {index}",
                "kind": "screw",
                "center_x": center_x,
                "center_y": center_y,
                "width": 0.047,
                "height": 0.035,
            }
        )
    items.extend(
        [
            {
                "id": "Clip 1",
                "kind": "clip",
                "center_x": 0.382,
                "center_y": 0.858,
                "width": 0.070,
                "height": 0.040,
            },
            {
                "id": "Clip 2",
                "kind": "clip",
                "center_x": 0.565,
                "center_y": 0.858,
                "width": 0.070,
                "height": 0.040,
            },
        ]
    )
    return items


def _save_upload(profile_root: Path, category: str, source_name: str, data: bytes) -> str:
    extension = Path(source_name).suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        extension = ".jpg"
    relative = Path("samples") / category / f"{uuid.uuid4().hex}{extension}"
    destination = profile_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return relative.as_posix()


def _read_profile_image(profile_root: Path, relative_path: str) -> np.ndarray:
    return read_image_bytes((profile_root / relative_path).read_bytes())


def _reference_image(profile_root: Path, profile: dict[str, Any]) -> np.ndarray:
    return _read_profile_image(profile_root, profile["reference_image"])


def create_profile(
    data_root: Path,
    requested_name: str,
    good_uploads: Iterable[tuple[str, bytes]],
) -> str:
    uploads = list(good_uploads)
    if not uploads:
        raise ValueError("Add at least one known-good part image before creating a profile.")
    name = safe_profile_name(requested_name)
    root = profile_dir(data_root, name)
    if root.exists():
        raise ValueError(f"A profile named '{name}' already exists.")
    root.mkdir(parents=True, exist_ok=False)

    good_samples: list[str] = []
    try:
        for original_name, image_data in uploads:
            # Validate before keeping it, so a bad upload cannot break training later.
            read_image_bytes(image_data)
            good_samples.append(_save_upload(root, "good", original_name, image_data))
        reference_relative = good_samples[0]
        reference = _read_profile_image(root, reference_relative)
        height, width = reference.shape[:2]
        profile: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "reference_image": reference_relative,
            "reference_size": [int(width), int(height)],
            "items": default_items(),
            "good_samples": good_samples,
            "defect_samples": [],
            "models": {},
        }
        save_profile(data_root, name, profile)
        rebuild_models(data_root, name)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return name


def add_good_samples(
    data_root: Path, profile_name: str, uploads: Iterable[tuple[str, bytes]]
) -> int:
    root = profile_dir(data_root, profile_name)
    profile = load_profile(data_root, profile_name)
    count = 0
    for original_name, image_data in uploads:
        read_image_bytes(image_data)
        profile["good_samples"].append(_save_upload(root, "good", original_name, image_data))
        count += 1
    if count:
        profile["updated_at"] = utc_now()
        save_profile(data_root, profile_name, profile)
        rebuild_models(data_root, profile_name)
    return count


def add_missing_samples(
    data_root: Path,
    profile_name: str,
    uploads: Iterable[tuple[str, bytes]],
    missing_item_ids: list[str],
) -> int:
    if not missing_item_ids:
        raise ValueError("Choose the item or items that are missing in these images.")
    root = profile_dir(data_root, profile_name)
    profile = load_profile(data_root, profile_name)
    valid_ids = {item["id"] for item in profile["items"]}
    invalid_ids = set(missing_item_ids) - valid_ids
    if invalid_ids:
        raise ValueError(f"Unknown item IDs: {', '.join(sorted(invalid_ids))}")
    count = 0
    for original_name, image_data in uploads:
        read_image_bytes(image_data)
        relative = _save_upload(root, "missing", original_name, image_data)
        profile["defect_samples"].append(
            {
                "image": relative,
                "missing_items": list(missing_item_ids),
                "added_at": utc_now(),
            }
        )
        count += 1
    if count:
        profile["updated_at"] = utc_now()
        save_profile(data_root, profile_name, profile)
        rebuild_models(data_root, profile_name)
    return count


def update_layout(data_root: Path, profile_name: str, items: list[dict[str, Any]]) -> None:
    if len(items) != 10:
        raise ValueError("This profile must contain exactly 10 inspection locations.")
    ids = [str(item.get("id", "")).strip() for item in items]
    if len(set(ids)) != 10 or any(not value for value in ids):
        raise ValueError("Each location must have a unique, non-empty ID.")
    normalized_items: list[dict[str, Any]] = []
    for item in items:
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in {"screw", "clip"}:
            raise ValueError("Each item type must be either 'screw' or 'clip'.")
        normalized = {"id": str(item["id"]).strip(), "kind": kind}
        for key in ("center_x", "center_y", "width", "height"):
            value = float(item[key])
            if not 0 < value <= 1:
                raise ValueError(f"{key} for {normalized['id']} must be between 0 and 1.")
            normalized[key] = round(value, 5)
        if normalized["width"] > 0.2 or normalized["height"] > 0.2:
            raise ValueError("Location rectangles should be smaller than 20% of the image.")
        normalized_items.append(normalized)
    profile = load_profile(data_root, profile_name)
    profile["items"] = normalized_items
    profile["updated_at"] = utc_now()
    save_profile(data_root, profile_name, profile)
    rebuild_models(data_root, profile_name)


def _alignment(source: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Register a camera image to the reference; resize only if registration fails."""
    reference_h, reference_w = reference.shape[:2]
    gray_source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=8)
    keypoints_source, descriptors_source = orb.detectAndCompute(gray_source, None)
    keypoints_reference, descriptors_reference = orb.detectAndCompute(gray_reference, None)
    if descriptors_source is not None and descriptors_reference is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(descriptors_source, descriptors_reference), key=lambda match: match.distance)
        matches = matches[:180]
        if len(matches) >= 12:
            source_points = np.float32([keypoints_source[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            reference_points = np.float32([keypoints_reference[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            homography, inlier_mask = cv2.findHomography(source_points, reference_points, cv2.RANSAC, 4.0)
            inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
            if homography is not None and inliers >= 10:
                aligned = cv2.warpPerspective(
                    source,
                    homography,
                    (reference_w, reference_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                return aligned, {
                    "method": "feature registration",
                    "inliers": inliers,
                    "matches": len(matches),
                    "reliable": inliers >= 18,
                }
    resized = cv2.resize(source, (reference_w, reference_h), interpolation=cv2.INTER_AREA)
    return resized, {"method": "resize fallback", "inliers": 0, "matches": 0, "reliable": False}


def _rect_for_item(item: dict[str, Any], image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    center_x = float(item["center_x"]) * width
    center_y = float(item["center_y"]) * height
    box_width = max(8, int(round(float(item["width"]) * width)))
    box_height = max(8, int(round(float(item["height"]) * height)))
    left = int(round(center_x - box_width / 2))
    top = int(round(center_y - box_height / 2))
    left = max(0, min(left, width - box_width))
    top = max(0, min(top, height - box_height))
    return left, top, min(width, left + box_width), min(height, top + box_height)


def _feature_vector(patch: np.ndarray, kind: str) -> np.ndarray:
    """Small, lighting-tolerant local feature vector for a single inspection site."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    small = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # Normalize local brightness so changes in lighting do not dominate the model.
    small = (small - small.mean()) / (small.std() + 0.05)
    histogram = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    histogram = histogram / (histogram.sum() + 1e-6)
    edges = cv2.Canny(gray, 50, 150)
    edge_grid = cv2.resize(edges, (6, 6), interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (92, 70, 45), (135, 255, 255))
    blue_fraction = float((blue_mask > 0).mean())
    bright_fraction = float((gray > 150).mean())
    dark_fraction = float((gray < 80).mean())
    color_summary = np.array(
        [
            float(hsv[:, :, 1].mean()) / 255.0,
            float(hsv[:, :, 2].mean()) / 255.0,
            blue_fraction * 5.0,
            bright_fraction,
            dark_fraction,
        ],
        dtype=np.float32,
    )
    # Blue coverage is especially informative for clips.  Screws are primarily
    # distinguished by the local silver head/ring and its texture.
    kind_marker = np.array([1.0 if kind == "clip" else 0.0], dtype=np.float32)
    return np.concatenate([small.flatten() * 0.22, histogram * 4.0, edge_grid, color_summary, kind_marker])


def _features_for_image(
    image: np.ndarray, reference: np.ndarray, items: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    aligned, alignment = _alignment(image, reference)
    features: dict[str, np.ndarray] = {}
    for item in items:
        left, top, right, bottom = _rect_for_item(item, aligned.shape)
        features[item["id"]] = _feature_vector(aligned[top:bottom, left:right], item["kind"])
    return features, alignment, aligned


def _model_from_vectors(vectors: list[np.ndarray]) -> dict[str, Any]:
    matrix = np.stack(vectors).astype(np.float32)
    mean = matrix.mean(axis=0)
    raw_scale = matrix.std(axis=0)
    non_zero = raw_scale[raw_scale > 1e-5]
    floor = float(np.median(non_zero) * 0.35) if non_zero.size else 0.04
    floor = max(0.025, min(floor, 0.15))
    scale = np.maximum(raw_scale, floor)
    distances = np.mean(np.abs((matrix - mean) / scale), axis=1)
    # Sparse calibration sets should be conservative: they create REVIEW instead
    # of an unjustified automatic rejection when a camera environment changes.
    if len(vectors) < 3:
        present_limit = 4.0
    else:
        present_limit = max(1.35, float(np.percentile(distances, 99)) + 0.30)
    return {
        "present_mean": mean.astype(float).tolist(),
        "present_scale": scale.astype(float).tolist(),
        "present_limit": round(float(present_limit), 4),
        "present_samples": len(vectors),
    }


def _distance(vector: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> float:
    return float(np.mean(np.abs((vector - mean) / (scale + 1e-6))))


def rebuild_models(data_root: Path, profile_name: str) -> dict[str, Any]:
    profile = load_profile(data_root, profile_name)
    root = profile_dir(data_root, profile_name)
    reference = _reference_image(root, profile)
    item_ids = [item["id"] for item in profile["items"]]
    present_vectors: dict[str, list[np.ndarray]] = {item_id: [] for item_id in item_ids}
    missing_vectors: dict[str, list[np.ndarray]] = {item_id: [] for item_id in item_ids}
    skipped: list[str] = []

    for relative_path in profile.get("good_samples", []):
        try:
            feature_map, _, _ = _features_for_image(_read_profile_image(root, relative_path), reference, profile["items"])
            for item_id in item_ids:
                present_vectors[item_id].append(feature_map[item_id])
        except Exception:
            skipped.append(relative_path)

    for sample in profile.get("defect_samples", []):
        try:
            feature_map, _, _ = _features_for_image(_read_profile_image(root, sample["image"]), reference, profile["items"])
            missing_ids = set(sample.get("missing_items", []))
            for item_id in item_ids:
                if item_id in missing_ids:
                    missing_vectors[item_id].append(feature_map[item_id])
                else:
                    # The other locations are known present and improve variation coverage.
                    present_vectors[item_id].append(feature_map[item_id])
        except Exception:
            skipped.append(sample.get("image", "unknown image"))

    models: dict[str, Any] = {}
    for item_id in item_ids:
        if not present_vectors[item_id]:
            raise ValueError(f"No usable known-good example was available for {item_id}.")
        model = _model_from_vectors(present_vectors[item_id])
        if missing_vectors[item_id]:
            model["missing_vectors"] = [vector.astype(float).tolist() for vector in missing_vectors[item_id]]
        else:
            model["missing_vectors"] = []
        models[item_id] = model
    profile["models"] = models
    profile["last_trained_at"] = utc_now()
    profile["training_warnings"] = skipped
    profile["updated_at"] = utc_now()
    save_profile(data_root, profile_name, profile)
    return profile


def inspect_image(data_root: Path, profile_name: str, image_data: bytes) -> dict[str, Any]:
    """Return local inspection results, a registered image, and an annotated image."""
    profile = load_profile(data_root, profile_name)
    if not profile.get("models"):
        profile = rebuild_models(data_root, profile_name)
    root = profile_dir(data_root, profile_name)
    source = read_image_bytes(image_data)
    reference = _reference_image(root, profile)
    feature_map, alignment, aligned = _features_for_image(source, reference, profile["items"])
    results: list[dict[str, Any]] = []
    for item in profile["items"]:
        item_id = item["id"]
        model = profile["models"].get(item_id)
        if not model:
            results.append({"id": item_id, "type": item["kind"], "status": "Review", "message": "Not calibrated"})
            continue
        vector = feature_map[item_id]
        present_mean = np.asarray(model["present_mean"], dtype=np.float32)
        present_scale = np.asarray(model["present_scale"], dtype=np.float32)
        present_distance = _distance(vector, present_mean, present_scale)
        present_limit = float(model["present_limit"])
        missing_vectors = [np.asarray(v, dtype=np.float32) for v in model.get("missing_vectors", [])]
        missing_distance: float | None = None
        if missing_vectors:
            missing_distance = min(_distance(vector, missing_vector, present_scale) for missing_vector in missing_vectors)
            # A labelled missing example is the most reliable basis for a decision.
            if missing_distance + 0.12 < present_distance:
                status = "Missing"
                confidence = min(0.99, 0.55 + (present_distance - missing_distance) / (present_distance + 0.5))
                message = "Matches labelled missing examples"
            elif present_distance <= present_limit:
                status = "Present"
                confidence = min(0.99, 0.55 + (missing_distance - present_distance) / (missing_distance + 0.5))
                message = "Matches good examples"
            else:
                status = "Review"
                confidence = 0.50
                message = "Does not match good or missing training examples closely"
        elif int(model.get("present_samples", 0)) < 3:
            status = "Review"
            confidence = 0.0
            message = "Need at least 3 good images for an automatic decision"
        elif present_distance > present_limit:
            # Until a real defect image is supplied, this is an anomaly-based
            # decision.  It remains visible in the UI as needing validation.
            status = "Missing"
            confidence = min(0.89, 0.50 + (present_distance - present_limit) / (present_distance + 0.5))
            message = "Different from the good-part reference; add a labelled missing example to validate"
        else:
            status = "Present"
            confidence = min(0.98, 0.65 + (present_limit - present_distance) / (present_limit + 0.5) * 0.30)
            message = "Matches good-part reference"
        results.append(
            {
                "id": item_id,
                "type": item["kind"],
                "status": status,
                "confidence": round(float(confidence) * 100, 1),
                "present_distance": round(present_distance, 3),
                "missing_distance": round(missing_distance, 3) if missing_distance is not None else None,
                "message": message,
            }
        )
    missing = [result["id"] for result in results if result["status"] == "Missing"]
    review = [result["id"] for result in results if result["status"] == "Review"]
    if missing:
        outcome = "FAIL"
    elif review or not alignment["reliable"]:
        outcome = "REVIEW"
    else:
        outcome = "PASS"
    annotated = draw_results(aligned, profile["items"], results, outcome)
    return {
        "outcome": outcome,
        "missing": missing,
        "review": review,
        "results": results,
        "alignment": alignment,
        "aligned_image": aligned,
        "annotated_image": annotated,
    }


def draw_layout_preview(image: np.ndarray, items: list[dict[str, Any]]) -> np.ndarray:
    preview = image.copy()
    for item in items:
        left, top, right, bottom = _rect_for_item(item, preview.shape)
        color = (255, 170, 0) if item["kind"] == "screw" else (255, 70, 0)
        cv2.rectangle(preview, (left, top), (right, bottom), color, 3)
        cv2.putText(preview, item["id"], (left, max(28, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return preview


def draw_results(
    aligned: np.ndarray,
    items: list[dict[str, Any]],
    results: list[dict[str, Any]],
    outcome: str,
) -> np.ndarray:
    output = aligned.copy()
    result_by_id = {result["id"]: result for result in results}
    colors = {"Present": (20, 180, 30), "Missing": (0, 0, 235), "Review": (0, 180, 255)}
    for item in items:
        result = result_by_id[item["id"]]
        status = result["status"]
        color = colors.get(status, (0, 180, 255))
        left, top, right, bottom = _rect_for_item(item, output.shape)
        cv2.rectangle(output, (left, top), (right, bottom), color, 4)
        label = f"{item['id']}: {status}"
        text_y = max(26, top - 8)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(output, (left, text_y - text_size[1] - 9), (left + text_size[0] + 10, text_y + 4), (20, 20, 20), -1)
        cv2.putText(output, label, (left + 5, text_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    banner_color = {"PASS": (25, 150, 35), "FAIL": (0, 0, 210), "REVIEW": (0, 150, 230)}[outcome]
    cv2.rectangle(output, (0, 0), (output.shape[1], 72), banner_color, -1)
    cv2.putText(output, f"QC RESULT: {outcome}", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    return output


def profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    defect_counts = {item["id"]: 0 for item in profile.get("items", [])}
    for sample in profile.get("defect_samples", []):
        for item_id in sample.get("missing_items", []):
            if item_id in defect_counts:
                defect_counts[item_id] += 1
    return {
        "good_images": len(profile.get("good_samples", [])),
        "missing_images": len(profile.get("defect_samples", [])),
        "defect_counts": defect_counts,
        "trained_at": profile.get("last_trained_at"),
        "warnings": profile.get("training_warnings", []),
    }
