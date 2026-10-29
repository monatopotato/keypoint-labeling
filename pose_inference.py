"""Overlapping crop inference with keypoints mapped to original image pixels."""
import math


def tile_windows(width, height):
    size = max(128, math.ceil(max(width, height) * 0.4))

    def starts(length):
        extent = min(size, length)
        count = math.ceil((length - extent) / (extent * 0.75)) + 1
        return [round(i * (length - extent) / max(1, count - 1)) for i in range(count)]

    return [(x, y, min(width, x + size), min(height, y + size))
            for y in starts(height) for x in starts(width)
            if min(width, size) < width or min(height, size) < height]


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def merge_candidates(candidates):
    def supported(candidate):
        if candidate['score'] >= 0.35:
            return True
        # Weak detections need a well-supported pose and agreement between views.
        # This is a conservative heuristic, not a guarantee of correctness.
        views = {other.get('view_id') for other in candidates
                 if box_iou(candidate['box'], other['box']) >= 0.45}
        return candidate['visible'] >= 12 and len(views) >= 2

    # Prefer confident detections with more usable joints; retain one complete
    # skeleton rather than averaging potentially different people's joints.
    ranked = sorted((c for c in candidates if supported(c)),
                    key=lambda c: c['score'] * (0.5 + 0.5 * c['visible'] / 17), reverse=True)
    kept = []
    for candidate in ranked:
        if not any(box_iou(candidate['box'], other['box']) >= 0.45 for other in kept):
            kept.append(candidate)
    return kept


def result_candidates(result, offset, dimensions, point_conf, minimum_joints=0):
    if result.keypoints is None:
        return []
    data = result.keypoints.data.cpu().tolist()
    boxes = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    width, height = dimensions
    ox, oy = offset
    candidates = []
    for points, box, score in zip(data, boxes, scores):
        if len(points) != 17 or not math.isfinite(score):
            continue
        keypoints = []
        for p in points:
            x, y = float(p[0]), float(p[1])
            confidence = float(p[2]) if len(p) > 2 else 1
            visible = (all(math.isfinite(v) for v in (x, y, confidence))
                       and confidence >= point_conf and x > 0 and y > 0)
            keypoints.append([round(min(width, max(0, x + ox)), 2),
                              round(min(height, max(0, y + oy)), 2), 2] if visible else [0, 0, 0])
        visible = sum(p[2] > 0 for p in keypoints)
        if visible < minimum_joints:
            continue
        candidates.append({'keypoints': keypoints, 'score': score, 'visible': visible,
                           'box': [box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy]})
    return candidates


def predict_distant(model, path, device, confidence, point_conf):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert('RGB')
    width, height = image.size
    windows = tile_windows(width, height)
    candidates = []
    # Full image preserves context and large people. Crops magnify small people.
    for view_id, window in enumerate([None, *windows]):
        crop = image if window is None else image.crop(window)
        result = model.predict(source=crop, conf=confidence, imgsz=1280,
                               device=device, verbose=False)[0]
        offset = (0, 0) if window is None else window[:2]
        found = result_candidates(result, offset, image.size, point_conf, minimum_joints=4)
        candidates.extend({**c, 'view_id': view_id} for c in found)
    kept = merge_candidates(candidates)
    people = [{'id': i + 1, 'source': 'yolo26', 'keypoints': c['keypoints']} for i, c in enumerate(kept)]
    return {'people': people, 'width': width, 'height': height, 'passes': len(windows) + 1}
