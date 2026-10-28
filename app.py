"""PoseFrame Studio — local-only COCO-17 keypoint annotation server."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import re
import shutil
import threading
import webbrowser
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("POSEFRAME_DATA", str(ROOT / "data"))).resolve()
UPLOAD_DIR = DATA / "frames"
PROJECT_FILE = DATA / "project.json"
CSV_FILE = DATA / "annotations.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODELS = {"yolo26x-pose.pt", "yolo26l-pose.pt", "yolo26m-pose.pt", "yolo26s-pose.pt", "yolo26n-pose.pt"}
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
CSV_FIELDS = (
    ["filename", "frame_index", "annotation_status", "person_id", "source", "num_keypoints", "image_width", "image_height"]
    + [f"{point}_{part}" for point in KEYPOINTS for part in ("x", "y", "v")]
)
PROJECT_LOCK = threading.RLock()
MODEL_LOCK = threading.Lock()

app = Flask(__name__, template_folder=".", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # per upload batch


def blank_project() -> dict[str, Any]:
    return {"folder": "", "mode": "", "frames": [], "annotations": {}}


def load_project() -> dict[str, Any]:
    if not PROJECT_FILE.exists():
        return blank_project()
    try:
        value = json.loads(PROJECT_FILE.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("frames"), list):
            value.setdefault("annotations", {})
            return value
    except (ValueError, OSError):
        pass
    return blank_project()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def write_project(project: dict[str, Any]) -> None:
    atomic_write(PROJECT_FILE, json.dumps(project, ensure_ascii=False, indent=2))


def filename_sort_key(name: str):
    """Timestamp names (2026-10-21-14-02-08-033, etc.) sort chronologically.

    Natural fallback sorts `frame_2.jpg` before `frame_10.jpg`.
    """
    stem = Path(name).stem
    digits = re.findall(r"\d+", stem)
    # Explicit YYYY-MM-DD-HH-MM-SS with optional milliseconds/fractional parts
    match = re.search(r"(?<!\d)(\d{4})[-_. ](\d{1,2})[-_. ](\d{1,2})[-_. T](\d{1,2})[-_. :](\d{1,2})[-_. :](\d{1,2})(?:[-_. ](\d+))?", stem)
    if match:
        nums = [int(x) if x is not None else 0 for x in match.groups()]
        return (0, tuple(nums), name.casefold())
    # Best-effort chronological ordering for filenames that use numeric timestamp blocks.
    parts = re.split(r"(\d+)", name.casefold())
    return (1, tuple((1, int(p)) if p.isdigit() else (0, p) for p in parts))


def scan_frames(folder: Path) -> list[str]:
    return sorted(
        (p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=filename_sort_key,
    )


def csv_rows(project: dict[str, Any]):
    for index, name in enumerate(project["frames"]):
        entry = project["annotations"].get(name) or {}
        people = entry.get("people", [])
        status = entry.get("status", "unlabeled")
        base = {
            "filename": name,
            "frame_index": index + 1,
            "annotation_status": status,
            "image_width": entry.get("width", ""),
            "image_height": entry.get("height", ""),
        }
        if not people:
            yield {**base, "person_id": "", "source": "", "num_keypoints": 0, **{f"{k}_{axis}": 0 for k in KEYPOINTS for axis in "xyv"}}
            continue
        for i, person in enumerate(people):
            points = person["keypoints"]
            row = {
                **base,
                "person_id": person.get("id", i + 1),
                "source": person.get("source", "manual"),
                "num_keypoints": sum(p[2] > 0 for p in points),
            }
            for label, (x, y, v) in zip(KEYPOINTS, points):
                row[f"{label}_x"] = x if v else 0
                row[f"{label}_y"] = y if v else 0
                row[f"{label}_v"] = v
            yield row


def write_csv(project: dict[str, Any]) -> None:
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = CSV_FILE.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows(project))
    os.replace(temp, CSV_FILE)


def active_folder(project: dict[str, Any]) -> Path:
    return Path(project["folder"]).resolve() if project.get("folder") else UPLOAD_DIR


def get_frame_path(project: dict[str, Any], filename: str) -> Path:
    if filename not in project.get("frames", []):
        abort(404, "Frame not in current project")
    folder = active_folder(project)
    file_path = (folder / filename).resolve()
    if file_path.parent != folder or not file_path.is_file():
        abort(404, "Frame file missing")
    return file_path


def snapshot(project: dict[str, Any]) -> dict[str, Any]:
    return {
        "folder": project.get("folder", ""),
        "mode": project.get("mode", ""),
        "count": len(project.get("frames", [])),
        "frames": [
            {
                "name": name,
                "status": (project["annotations"].get(name) or {}).get("status", "unlabeled"),
                "people": len((project["annotations"].get(name) or {}).get("people", [])),
            }
            for name in project["frames"]
        ],
    }


def validate_people(raw: Any, width: int, height: int) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) > 50:
        raise ValueError("Expected at most 50 person annotations")
    people = []
    for i, person in enumerate(raw):
        if not isinstance(person, dict) or not isinstance(person.get("keypoints"), list) or len(person["keypoints"]) != 17:
            raise ValueError(f"Person {i + 1} must contain exactly 17 keypoints")
        points = []
        for point in person["keypoints"]:
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError("Each keypoint must be [x, y, visibility]")
            x, y, v = point
            if isinstance(v, bool) or v not in (0, 1, 2):
                raise ValueError("Visibility must be 0, 1 or 2")
            try:
                x, y = float(x), float(y)
            except (ValueError, TypeError):
                raise ValueError("Keypoint coordinates must be numbers") from None
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("Coordinates must be finite")
            if v == 0:
                points.append([0, 0, 0])
            else:
                points.append([round(min(max(x, 0), width), 2), round(min(max(y, 0), height), 2), int(v)])
        people.append({
            "id": i + 1,
            "source": "yolo26" if person.get("source") == "yolo26" else "manual",
            "keypoints": points,
        })
    return people


@app.get("/")
def index():
    return send_file(ROOT / "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "ultralytics_installed": importlib.util.find_spec("ultralytics") is not None, "csv_path": str(CSV_FILE)})


@app.get("/api/project")
def project_details():
    with PROJECT_LOCK:
        return jsonify(snapshot(load_project()))


@app.post("/api/import/start")
def import_start():
    with PROJECT_LOCK:
        # The browser confirms before calling this. Reset the local working copy.
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        project = {"folder": str(UPLOAD_DIR), "mode": "upload", "frames": [], "annotations": {}}
        write_project(project)
        write_csv(project)
    return jsonify({"ok": True})


@app.post("/api/import/batch")
def import_batch():
    uploaded = request.files.getlist("files")
    if not uploaded:
        return jsonify({"error": "No files provided"}), 400
    if len(uploaded) > 75:
        return jsonify({"error": "Batch exceeds 75 files"}), 400
    with PROJECT_LOCK:
        project = load_project()
        if project.get("mode") != "upload":
            return jsonify({"error": "Start a folder import first"}), 409
        saved = 0
        for file in uploaded:
            # Preserve original frame names (including spaces / Unicode).
            # Strip any directory traversal and platform separators.
            name = Path((file.filename or "").replace("\\", "/")).name
            name = re.sub(r"[\x00-\x1f]", "_", name)
            if not name or Path(name).suffix.lower() not in IMAGE_EXTS:
                continue
            dest = UPLOAD_DIR / name
            stem, suffix, n = dest.stem, dest.suffix, 2
            while dest.exists():
                dest = UPLOAD_DIR / f"{stem}_{n}{suffix}"
                n += 1
            file.save(dest)
            saved += 1
    return jsonify({"saved": saved})


@app.post("/api/import/finish")
def import_finish():
    with PROJECT_LOCK:
        project = load_project()
        if project.get("mode") != "upload":
            return jsonify({"error": "Start an upload first"}), 409
        project["frames"] = scan_frames(UPLOAD_DIR)
        write_project(project)
        write_csv(project)
        return jsonify(snapshot(project))


@app.post("/api/import/path")
def import_existing_path():
    body = request.get_json(silent=True) or {}
    folder_text = str(body.get("path", "")).strip().strip('"').strip("'")
    folder = Path(folder_text).expanduser().resolve() if folder_text else None
    if not folder or not folder.is_dir():
        return jsonify({"error": "Folder does not exist on the computer running the server."}), 400
    try:
        names = scan_frames(folder)
    except OSError as e:
        return jsonify({"error": str(e)}), 400
    if not names:
        return jsonify({"error": "No JPG, PNG, WEBP or BMP frames found directly in that folder."}), 400
    with PROJECT_LOCK:
        old = load_project()
        # Preserve existing annotations if user opens the very same folder again.
        same = old.get("mode") == "path" and old.get("folder") == str(folder)
        project = {
            "mode": "path", "folder": str(folder), "frames": names,
            "annotations": {k: v for k, v in old.get("annotations", {}).items() if k in names} if same else {},
        }
        write_project(project)
        write_csv(project)
        return jsonify(snapshot(project))


@app.get("/api/image/<path:filename>")
def serve_image(filename: str):
    with PROJECT_LOCK:
        file_path = get_frame_path(load_project(), filename)
    return send_file(file_path, conditional=True)


@app.get("/api/annotations/<path:filename>")
def get_annotations(filename: str):
    with PROJECT_LOCK:
        project = load_project()
        get_frame_path(project, filename)
        entry = project["annotations"].get(filename, {})
        return jsonify({"people": entry.get("people", []), "status": entry.get("status", "unlabeled")})


@app.put("/api/annotations/<path:filename>")
def save_annotations(filename: str):
    body = request.get_json(silent=True) or {}
    try:
        width = int(body.get("width", 0))
        height = int(body.get("height", 0))
        if not (1 <= width <= 100000 and 1 <= height <= 100000):
            raise ValueError("Invalid image dimensions")
        people = validate_people(body.get("people", []), width, height)
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    with PROJECT_LOCK:
        project = load_project()
        get_frame_path(project, filename)
        project["annotations"][filename] = {
            "status": "labeled" if people else "reviewed_empty",
            "width": width,
            "height": height,
            "people": people,
        }
        write_project(project)
        write_csv(project)
        return jsonify({"ok": True, "people": len(people), "status": project["annotations"][filename]["status"]})


@lru_cache(maxsize=3)
def load_yolo_model(model_name: str):
    from ultralytics import YOLO
    return YOLO(model_name)


@app.post("/api/predict/<path:filename>")
def predict(filename: str):
    body = request.get_json(silent=True) or {}
    model_name = str(body.get("model", "yolo26x-pose.pt"))
    if model_name not in MODELS:
        return jsonify({"error": "Unsupported pose model"}), 400
    try:
        min_conf = float(body.get("confidence", 0.25))
        min_point_conf = float(body.get("keypoint_confidence", 0.45))
        if not (0 <= min_conf <= 1 and 0 <= min_point_conf <= 1):
            raise ValueError()
    except (ValueError, TypeError):
        return jsonify({"error": "Confidence thresholds must be between 0 and 1"}), 400
    with PROJECT_LOCK:
        path = get_frame_path(load_project(), filename)
    if importlib.util.find_spec("ultralytics") is None:
        return jsonify({"error": "Ultralytics is not installed. Run install_yolo.bat (Windows) or ./install_yolo.sh (macOS/Linux), then retry."}), 503
    try:
        # Model loads once, on first use. We serialize predictions to avoid
        # racing model initialization or CUDA kernels in this local server.
        with MODEL_LOCK:
            model = load_yolo_model(model_name)
            result = model.predict(source=str(path), conf=min_conf, imgsz=640, verbose=False)[0]
            data = result.keypoints.data.cpu().tolist() if result.keypoints is not None else []
        people = []
        for points in data:
            if len(points) != 17:
                continue
            out = []
            for p in points:
                x, y = float(p[0]), float(p[1])
                confidence = float(p[2]) if len(p) > 2 else 1
                # YOLO keypoint 3rd dimension is *confidence*, not COCO visibility.
                # COCO v=1 (occluded) cannot be inferred reliably from confidence.
                visible = 2 if confidence >= min_point_conf and x > 0 and y > 0 else 0
                out.append([round(x, 2) if visible else 0, round(y, 2) if visible else 0, visible])
            people.append({"id": len(people) + 1, "source": "yolo26", "keypoints": out})
        return jsonify({"people": people, "model": model_name})
    except Exception as exc:
        return jsonify({"error": f"Pose inference failed: {exc}"}), 500


@app.get("/api/export")
def export_csv():
    with PROJECT_LOCK:
        if not CSV_FILE.exists():
            write_csv(load_project())
    return send_file(CSV_FILE, as_attachment=True, download_name="coco_keypoints.csv", mimetype="text/csv")


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Upload batch is too large. Try smaller image files or a local folder path."}), 413


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("POSEFRAME_HOST", "127.0.0.1")
    port = int(os.environ.get("POSEFRAME_PORT", "8765"))
    if os.environ.get("POSEFRAME_OPEN_BROWSER", "1") == "1":
        threading.Timer(1.1, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    print(f"\n  PoseFrame Studio  →  http://{host}:{port}")
    print(f"  Annotations CSV   →  {CSV_FILE}\n")
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
