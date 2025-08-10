import os
import json
import csv
import numpy as np
import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

INPUT_PATH = "test.mp4"
OUTPUT_PATH = "output.mp4"
FACE_DB_JSON = "face_db.json"
LOG_CSV = "face_log.csv"

SIMILARITY_THRESHOLD = 0.4  # Cosine similarity cutoff (higher is more similar)
EMBEDDING_BUFFER_SIZE = 20
FACE_DETECT_INTERVAL = 2
PROCESS_FPS = 20
MIN_TRACK_SECONDS = 1.0

# Initialize models
yolo = YOLO("yolo11s.pt")
face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(320, 320))

def normalize_vec(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return None
    return v / norm

# --- Load or init face DB ---
face_db = {}
next_face_id = 0

if os.path.exists(FACE_DB_JSON):
    with open(FACE_DB_JSON, 'r') as f:
        raw = json.load(f)
    face_db = {int(k): [np.array(e, dtype=np.float32) for e in v] for k, v in raw.items()}
    next_face_id = max(face_db.keys(), default=-1) + 1

# --- Build FAISS index from face_db ---
embedding_dim = None
pos_to_fid = []
faiss_index = None

def build_faiss():
    global faiss_index, pos_to_fid, embedding_dim
    pos_to_fid.clear()
    all_vecs = []
    for fid in sorted(face_db.keys()):
        for emb in face_db[fid]:
            norm_emb = normalize_vec(emb)
            if norm_emb is not None:
                all_vecs.append(norm_emb)
                pos_to_fid.append(fid)
    if not all_vecs:
        faiss_index = None
        embedding_dim = None
        return
    embedding_dim = all_vecs[0].shape[0]
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(np.vstack(all_vecs).astype('float32'))

build_faiss()

# Video capture and output
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {INPUT_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
skip_frames = max(1, int(fps / PROCESS_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, PROCESS_FPS, (width, height))

frame_count = 0
track_to_face = {}
pending_tracks = {}
face_log = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % skip_frames != 0:
        frame_count += 1
        continue

    # YOLO person detection + tracking
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

    # Face detection every FACE_DETECT_INTERVAL frames
    cached_faces = []
    if frame_count % FACE_DETECT_INTERVAL == 0:
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
                print("[WARN] InsightFace error:", e)
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

    for embedding, bbox, track_id in cached_faces:
        norm_emb = normalize_vec(embedding)
        if norm_emb is None:
            continue

        matched_id = None

        if track_id not in track_to_face:
            # Wait minimum tracked time before matching
            if track_id in pending_tracks:
                duration = (frame_count - pending_tracks[track_id]['start_frame']) / fps
                if duration < MIN_TRACK_SECONDS:
                    continue

            # FAISS search
            if FAISS_AVAILABLE and faiss_index is not None and faiss_index.ntotal > 0:
                D, I = faiss_index.search(norm_emb[np.newaxis, :].astype('float32'), 1)
                sim = float(D[0][0])
                pos = int(I[0][0])
                if 0 <= pos < len(pos_to_fid):
                    fid_candidate = pos_to_fid[pos]
                    if sim >= SIMILARITY_THRESHOLD:
                        matched_id = fid_candidate

            # Brute force fallback
            if matched_id is None:
                best_sim = -1
                best_fid = None
                for fid, embs in face_db.items():
                    for e in embs:
                        sim = np.dot(norm_emb, e)
                        if sim > best_sim:
                            best_sim = sim
                            best_fid = fid
                if best_sim >= SIMILARITY_THRESHOLD:
                    matched_id = best_fid

            # New face
            if matched_id is None:
                matched_id = next_face_id
                next_face_id += 1
                face_db[matched_id] = []

            track_to_face[track_id] = matched_id
            if track_id in pending_tracks:
                del pending_tracks[track_id]

        else:
            matched_id = track_to_face[track_id]

        # Add embedding & maintain buffer
        face_db.setdefault(matched_id, [])
        face_db[matched_id].append(norm_emb)
        if len(face_db[matched_id]) > EMBEDDING_BUFFER_SIZE:
            face_db[matched_id].pop(0)

        # Rebuild FAISS index to include new embeddings (periodically or here for demo)
        build_faiss()

        current_faces.append((matched_id, bbox))

        # Log appearances
        if matched_id not in face_log:
            face_log[matched_id] = []
        if not face_log[matched_id] or frame_count - face_log[matched_id][-1][1] > FACE_DETECT_INTERVAL * 2:
            face_log[matched_id].append([frame_count, frame_count])
        else:
            face_log[matched_id][-1][1] = frame_count

    # Draw boxes and IDs
    for track_id, px1, py1, px2, py2 in person_boxes:
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
    for fid, bbox in current_faces:
        fx1, fy1, fx2, fy2 = bbox
        if fx1 is not None:
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
            cv2.putText(frame, f"FaceID: {fid}", (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(frame, f"Time: {frame_count / fps:.2f}s", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Save persistent face DB
with open(FACE_DB_JSON, 'w') as f:
    serial = {str(fid): [e.tolist() for e in embs] for fid, embs in face_db.items()}
    json.dump(serial, f, indent=2)

# Append CSV logs (with run timestamp)
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

from datetime import datetime
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
max_gap_frames = int(5 * fps)

file_exists = os.path.exists(LOG_CSV)
with open(LOG_CSV, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["FaceID", "StartTime", "EndTime", "DurationSeconds", "RunTimestamp"])
    for face_id, segments in face_log.items():
        merged = merge_segments(segments, max_gap_frames)
        for start, end in merged:
            writer.writerow([
                face_id,
                f"{start / fps:.2f}",
                f"{end / fps:.2f}",
                f"{(end - start) / fps:.2f}",
                run_timestamp
            ])
