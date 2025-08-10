import os
import json
import csv
import numpy as np
import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
INPUT_PATH = "test.mp4"
OUTPUT_PATH = "output.mp4"
LOG_CSV = "face_log.csv"
PERSIST_JSON = "face_db.json"

SIMILARITY_THRESHOLD = 0.5
EMBEDDING_BUFFER_SIZE = 10
FACE_DETECT_INTERVAL = 2
PROCESS_FPS = 20
MIN_TRACK_SECONDS = 1.0

# ----------------------------------------

# Initialize models
yolo = YOLO("yolo11s.pt")
face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(320, 320))

# Load or init face DB
if os.path.exists(PERSIST_JSON):
    with open(PERSIST_JSON, 'r') as f:
        raw_db = json.load(f)
    face_db = {int(k): [np.array(e, dtype=np.float32) for e in v] for k, v in raw_db.items()}
    next_face_id = max(face_db.keys()) + 1 if face_db else 0
else:
    face_db = {}
    next_face_id = 0

def normalize_vec(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return None
    return v / norm

def find_matching_face(norm_emb):
    best_id = None
    best_dist = 1.0  # cosine distance ranges from 0 (same) to 2 (opposite)
    for fid, embeddings in face_db.items():
        for e in embeddings:
            e_norm = normalize_vec(e)
            if e_norm is None:
                continue
            dist = 1 - np.dot(norm_emb, e_norm)  # cosine distance = 1 - cosine similarity
            if dist < best_dist:
                best_dist = dist
                best_id = fid
    if best_dist < SIMILARITY_THRESHOLD:
        return best_id
    return None

def add_embedding(fid, emb):
    if fid not in face_db:
        face_db[fid] = []
    face_db[fid].append(emb)
    # Keep buffer size fixed
    if len(face_db[fid]) > EMBEDDING_BUFFER_SIZE:
        face_db[fid].pop(0)

def merge_segments(segments, max_gap_frames):
    if not segments:
        return []
    segments.sort()
    merged = [segments[0][:]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap_frames:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return merged

# Video capture and writer setup
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {INPUT_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
skip_frames = max(1, int(fps / PROCESS_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, PROCESS_FPS, (width, height))

# Tracking structures
track_to_face = {}
pending_tracks = {}
face_log = {}

frame_count = 0
cached_faces = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % skip_frames != 0:
        frame_count += 1
        continue

    # Person detection + tracking
    results = yolo.track(frame, persist=True, classes=[0], verbose=False, imgsz=320)[0]
    person_boxes = []
    if results.boxes is not None:
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            px1, py1, px2, py2 = coords
            track_id = int(box.id.item()) if box.id is not None else None
            if track_id is not None:
                px1 = max(0, min(px1, width - 1))
                py1 = max(0, min(py1, height - 1))
                px2 = max(0, min(px2, width))
                py2 = max(0, min(py2, height))
                if px2 <= px1 or py2 <= py1:
                    continue
                person_boxes.append((track_id, px1, py1, px2, py2))
                if track_id not in pending_tracks and track_id not in track_to_face:
                    pending_tracks[track_id] = {'start_frame': frame_count}

    # Face detection every N frames
    if frame_count % FACE_DETECT_INTERVAL == 0:
        cached_faces.clear()
        for track_id, px1, py1, px2, py2 in person_boxes:
            face_h = int((py2 - py1) * 0.8)
            cy1 = py1
            cy2 = py1 + face_h
            cy1 = max(0, cy1)
            cy2 = min(height, cy2)
            if cy2 <= cy1 or px2 <= px1:
                continue
            face_crop = frame[cy1:cy2, px1:px2]
            if face_crop.size == 0:
                continue
            try:
                faces = face_app.get(face_crop)
            except Exception as e:
                print("[WARN] insightface error:", e)
                continue
            if not faces:
                continue
            f = faces[0]
            emb = getattr(f, "embedding", None)
            bbox = getattr(f, "bbox", None)
            if emb is None or bbox is None:
                continue
            emb = np.asarray(emb, dtype=np.float32)
            if emb.size == 0:
                continue
            bx1, by1, bx2, by2 = bbox.astype(int)
            fx1 = px1 + bx1
            fy1 = cy1 + by1
            fx2 = px1 + bx2
            fy2 = cy1 + by2
            fx1 = max(0, min(fx1, width - 1))
            fy1 = max(0, min(fy1, height - 1))
            fx2 = max(0, min(fx2, width))
            fy2 = max(0, min(fy2, height))
            if fx2 <= fx1 or fy2 <= fy1:
                cached_faces.append((emb, (None, None, None, None), track_id))
            else:
                cached_faces.append((emb, (fx1, fy1, fx2, fy2), track_id))

    current_faces = []

    # Match embeddings & assign face IDs
    for embedding, bbox, track_id in cached_faces:
        norm_emb = normalize_vec(embedding)
        if norm_emb is None:
            continue

        matched_id = None

        if track_id not in track_to_face:
            # Wait for minimum track duration before assigning face_id
            if track_id in pending_tracks:
                duration = (frame_count - pending_tracks[track_id]['start_frame']) / fps
                if duration < MIN_TRACK_SECONDS:
                    continue

            matched_id = find_matching_face(norm_emb)

            if matched_id is None:
                matched_id = next_face_id
                next_face_id += 1

            add_embedding(matched_id, embedding)
            track_to_face[track_id] = matched_id
            if track_id in pending_tracks:
                del pending_tracks[track_id]
        else:
            matched_id = track_to_face[track_id]
            add_embedding(matched_id, embedding)

        current_faces.append((matched_id, bbox))

        # Log appearance segments for CSV
        if matched_id not in face_log:
            face_log[matched_id] = []
        if not face_log[matched_id] or frame_count - face_log[matched_id][-1][1] > FACE_DETECT_INTERVAL * 2:
            face_log[matched_id].append([frame_count, frame_count])
        else:
            face_log[matched_id][-1][1] = frame_count

    # Draw boxes on frame
    for track_id, px1, py1, px2, py2 in person_boxes:
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

    for fid, bbox in current_faces:
        fx1, fy1, fx2, fy2 = bbox
        if fx1 is not None:
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
            cv2.putText(frame, f"FaceID: {fid}", (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    elapsed_seconds = frame_count / fps
    cv2.putText(frame, f"Elapsed: {elapsed_seconds:.2f}s", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("YOLO + InsightFace", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Save persistent face_db
with open(PERSIST_JSON, 'w') as f:
    serial = {str(fid): [e.tolist() for e in embs] for fid, embs in face_db.items()}
    json.dump(serial, f, indent=2)

# Append to CSV log, create header if file doesn't exist
max_gap_seconds = 5
max_gap_frames = int(max_gap_seconds * fps)

file_exists = os.path.exists(LOG_CSV)
with open(LOG_CSV, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["FaceID", "StartTime", "EndTime", "DurationSeconds"])

    for face_id, segments in face_log.items():
        merged_segments = merge_segments(segments, max_gap_frames)
        total_duration = sum((end - start) / fps for start, end in merged_segments)
        if total_duration > 1.0:
            for start, end in merged_segments:
                writer.writerow([
                    face_id,
                    f"{start / fps:.2f}",
                    f"{end / fps:.2f}",
                    f"{(end - start) / fps:.2f}"
                ])
