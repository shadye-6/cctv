import os
import json
import time
import csv
import numpy as np
import cv2
import pandas as pd

# models
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# optional speedup
try:
    import faiss
    FAISS_AVAILABLE = True
    print("[INFO] FAISS is available.")
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARN] FAISS not found. Falling back to NumPy. For faster performance, install faiss-cpu or faiss-gpu.")


# ---------------- CONFIG ----------------
INPUT_PATH = "test.mp4"
OUTPUT_PATH = "output.mp4"
LOG_CSV = "face_log.csv"
PERSIST_JSON = "face_db.json"
PERSIST_INDEX = "face_index.faiss" # Note: This will be overwritten if it exists, as we rebuild on start

# hyperparams
# A lower threshold means stricter matching. 0.4 is a good starting point.
SIMILARITY_THRESHOLD = 0.4
EMBEDDING_BUFFER_SIZE = 10    # Number of recent embeddings to keep per person
FACE_DETECT_INTERVAL = 10     # Run face detection every 10 processed frames
PROCESS_FPS = 10              # Process video at 10 FPS
MIN_TRACK_SECONDS = 1.0       # Track a person for at least 1s before assigning ID

# InsightFace embedding dim (will autodetect on first embedding)
EMBEDDING_DIM = None
# ----------------------------------------

# ---------------- models init ----------------
print("[INFO] Initializing models...")
yolo = YOLO("yolov8n.pt")  # Using a standard, smaller YOLO model
face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(320, 320)) # Use ctx_id=0 for GPU if available, -1 for CPU
print("[INFO] Models initialized.")
# ------------------------------------------------

# ---------------- video init ----------------
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {INPUT_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
skip_frames = max(1, round(fps / PROCESS_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, PROCESS_FPS, (width, height))
# ----------------------------------------------

# ---------------- in-memory structures ----------------
face_db = {}          # {face_id(int): [embedding(np.array float32), ...]}
next_face_id = 0
track_to_face = {}    # {track_id: face_id}
pending_tracks = {}   # {track_id: {'start_frame': frame_idx}}
face_log = {}         # {face_id: [[start_frame, end_frame], ...]}

active_tracks = {}    # {track_id: {'face_id': fid, 'bbox': (x1,y1,x2,y2), 'last_seen': frame_count}}

persistent_index = None
persistent_pos_to_fid = []   # list where index position i -> face_id
# -------------------------------------------------------

def normalize_vec(v):
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v # Return zero vector if norm is zero
    return (v / norm).astype(np.float32)

def rebuild_persistent_index():
    global persistent_index, persistent_pos_to_fid, EMBEDDING_DIM
    if not FAISS_AVAILABLE:
        persistent_index = None
        persistent_pos_to_fid = []
        return

    print("[INFO] Rebuilding FAISS index from face_db...")
    all_vecs = []
    pos_map = []
    for fid in sorted(face_db.keys()):
        for emb in face_db[fid]:
            # Always normalize before adding to index
            norm_emb = normalize_vec(emb)
            all_vecs.append(norm_emb)
            pos_map.append(fid)

    if not all_vecs:
        persistent_index = None
        persistent_pos_to_fid = []
        print("[INFO] No embeddings to build index.")
        return

    if EMBEDDING_DIM is None:
        EMBEDDING_DIM = all_vecs[0].shape[0]

    persistent_index = faiss.IndexFlatIP(EMBEDDING_DIM) # IP = Inner Product (Cosine Similarity for normalized vectors)
    arr = np.vstack(all_vecs).astype('float32')
    persistent_index.add(arr)
    persistent_pos_to_fid = pos_map
    print(f"[INFO] FAISS index rebuilt. Contains {persistent_index.ntotal} vectors.")

def load_persistent():
    global face_db, next_face_id, EMBEDDING_DIM
    if os.path.exists(PERSIST_JSON):
        print(f"[INFO] Loading face database from {PERSIST_JSON}...")
        with open(PERSIST_JSON, 'r') as f:
            raw = json.load(f)
        face_db = {int(k): [np.array(e, dtype=np.float32) for e in v] for k, v in raw.items()}
        if face_db:
            next_face_id = max(face_db.keys()) + 1
            # Auto-detect embedding dimension from loaded data
            for v in face_db.values():
                if v:
                    EMBEDDING_DIM = v[0].shape[0]
                    break
        else:
            next_face_id = 0
        print(f"[INFO] Loaded {len(face_db)} known faces.")
    else:
        print("[INFO] No existing face database found.")
        face_db = {}
        next_face_id = 0

    # Always rebuild the index from the loaded DB to ensure synchronization
    rebuild_persistent_index()

def save_persistent():
    print("[INFO] Saving data...")
    serial = {str(fid): [emb.tolist() for emb in embs] for fid, embs in face_db.items()}
    with open(PERSIST_JSON, 'w') as f:
        json.dump(serial, f, indent=2)

    # Saving the FAISS index is optional, as we rebuild it on start
    if FAISS_AVAILABLE and persistent_index is not None and persistent_index.ntotal > 0:
        try:
            faiss.write_index(persistent_index, PERSIST_INDEX)
            print(f"[INFO] FAISS index saved to {PERSIST_INDEX}")
        except Exception as e:
            print(f"[WARN] Failed to write FAISS index: {e}")

def load_existing_logs(csv_path):
    if not os.path.exists(csv_path):
        return {}
    if os.path.getsize(csv_path) == 0:
        return {}
    df = pd.read_csv(csv_path)
    logs = {}
    for _, row in df.iterrows():
        fid = int(row["FaceID"])
        start_frame = int(float(row["StartTime"]) * fps)
        end_frame = int(float(row["EndTime"]) * fps)
        if fid not in logs:
            logs[fid] = []
        logs[fid].append([start_frame, end_frame])
    return logs

def find_matching_face(norm_emb):
    if not face_db:
        return None

    query_vector = norm_emb[np.newaxis, :].astype('float32')

    # FAISS search path
    if FAISS_AVAILABLE and persistent_index is not None and persistent_index.ntotal > 0:
        try:
            # Search for the single nearest neighbor
            distances, indices = persistent_index.search(query_vector, 1)
            best_sim = distances[0][0]
            best_pos = indices[0][0]

            cosine_distance = 1.0 - best_sim
            if cosine_distance < SIMILARITY_THRESHOLD:
                return persistent_pos_to_fid[best_pos]
        except Exception as e:
            print(f"[WARN] FAISS search failed: {e}. Falling back to NumPy.")
            # Fallback to numpy if FAISS fails for some reason
            pass

    # Fallback NumPy search path (also used if FAISS fails)
    all_embs = []
    all_ids = []
    for fid, embs in face_db.items():
        for e in embs:
            all_embs.append(normalize_vec(e))
            all_ids.append(fid)

    if not all_embs:
        return None

    A = np.vstack(all_embs)
    sims = np.dot(A, query_vector.T).flatten()
    best_idx = np.argmax(sims)
    best_sim = sims[best_idx]
    cosine_distance = 1.0 - best_sim

    if cosine_distance < SIMILARITY_THRESHOLD:
        return all_ids[best_idx]

    return None

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

# ---- startup ----
load_persistent()
face_log = load_existing_logs(LOG_CSV)

frame_count = 0
total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# ----------- MAIN LOOP --------------
print("[INFO] Starting video processing...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # To ensure we process at 'PROCESS_FPS', we skip frames
    if frame_count % skip_frames != 0:
        frame_count += 1
        continue
    
    # YOLO tracking (persons only)
    results = yolo.track(frame, persist=True, classes=[0], verbose=False)[0]
    person_boxes = []
    detected_track_ids = set()

    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)

        for i, track_id in enumerate(track_ids):
            detected_track_ids.add(track_id)
            x1, y1, x2, y2 = boxes[i]
            person_boxes.append((track_id, x1, y1, x2, y2))

            if track_id not in pending_tracks and track_id not in track_to_face:
                pending_tracks[track_id] = {'start_frame': frame_count}

    # Face detection and Re-ID logic, runs periodically
    if frame_count % (skip_frames * FACE_DETECT_INTERVAL) == 0:
        should_rebuild_index = False
        for track_id, px1, py1, px2, py2 in person_boxes:
            person_crop = frame[py1:py2, px1:px2]
            if person_crop.size == 0:
                continue

            try:
                # InsightFace is good at finding faces in larger crops
                faces = face_app.get(person_crop)
                if not faces:
                    continue
                
                # Assume the largest face in the crop is the correct one
                face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
                
                emb = getattr(face, "embedding", None)
                if emb is None:
                    continue
                
                norm_emb = normalize_vec(emb)
                
                # Check if this track is already identified
                if track_id in track_to_face:
                    matched_id = track_to_face[track_id]
                else:
                    # Check if this person has been tracked long enough
                    if track_id in pending_tracks:
                        tracked_duration = (frame_count - pending_tracks[track_id]['start_frame']) / fps
                        if tracked_duration < MIN_TRACK_SECONDS:
                            continue # Not tracked long enough, skip for now

                    matched_id = find_matching_face(norm_emb)

                    if matched_id is None:
                        # No match found, create a new ID
                        matched_id = next_face_id
                        face_db[matched_id] = []
                        face_log[matched_id] = []
                        next_face_id += 1
                        print(f"[NEW] New face detected. Assigning FaceID: {matched_id}")
                    
                    track_to_face[track_id] = matched_id
                    if track_id in pending_tracks:
                        del pending_tracks[track_id]

                # Update the face_db buffer
                if len(face_db[matched_id]) >= EMBEDDING_BUFFER_SIZE:
                    face_db[matched_id].pop(0)
                face_db[matched_id].append(emb)
                should_rebuild_index = True

                # Update active track info for drawing
                bx1, by1, bx2, by2 = face.bbox.astype(int)
                fx1, fy1, fx2, fy2 = px1 + bx1, py1 + by1, px1 + bx2, py1 + by2
                active_tracks[track_id] = {'face_id': matched_id, 'bbox': (fx1, fy1, fx2, fy2), 'last_seen': frame_count}

                # Update appearance log
                if not face_log[matched_id] or frame_count > face_log[matched_id][-1][1] + (skip_frames * FACE_DETECT_INTERVAL * 2):
                    face_log[matched_id].append([frame_count, frame_count])
                else:
                    face_log[matched_id][-1][1] = frame_count
            
            except Exception as e:
                print(f"[WARN] Error processing track {track_id}: {e}")
                continue
        
        # After processing all faces in the frame, rebuild the index if needed
        if should_rebuild_index:
            rebuild_persistent_index()

    # --- Drawing Logic (to prevent flickering) ---
    # Update active_tracks with person boxes for tracks that didn't get a new face detection
    for track_id, px1, py1, px2, py2 in person_boxes:
        if track_id in track_to_face:
            if track_id not in active_tracks:
                 active_tracks[track_id] = {'face_id': track_to_face[track_id], 'bbox': None, 'last_seen': frame_count}
            else:
                 active_tracks[track_id]['last_seen'] = frame_count

    # Draw boxes for all currently active tracks
    for track_id, data in list(active_tracks.items()):
        # Draw green person box
        for tid, px1, py1, px2, py2 in person_boxes:
            if tid == track_id:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                break
        
        # Draw red face box and label
        fid = data['face_id']
        face_bbox = data.get('bbox')
        if face_bbox:
            fx1, fy1, fx2, fy2 = face_bbox
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
            label_pos = (fx1, fy1 - 10 if fy1 > 20 else fy1 + 20)
            cv2.putText(frame, f"FaceID: {fid}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Clean up old tracks that are no longer detected
    for track_id in list(active_tracks.keys()):
        if track_id not in detected_track_ids:
            del active_tracks[track_id]

    # --- Display Info & Write Frame ---
    # Display progress
    progress = (frame_count / total_frames_in_video) * 100 if total_frames_in_video > 0 else 0
    elapsed_seconds = frame_count / fps
    cv2.putText(frame, f"Time: {elapsed_seconds:.1f}s | Progress: {progress:.1f}%", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("YOLO + InsightFace Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1 # Increment by 1, as we check against this raw count

# ---- cleanup and save ----
print("[INFO] Processing finished. Cleaning up...")
cap.release()
out.release()
cv2.destroyAllWindows()

save_persistent()

# WRITE CSV log (overwrite with merged data)
with open(LOG_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["FaceID", "StartTime", "EndTime", "DurationSeconds"])

    max_gap_seconds = 5.0 # Merge appearances if they are within 5 seconds of each other
    max_gap_frames = int(max_gap_seconds * fps)

    for face_id, spans in sorted(face_log.items()):
        if not spans: continue
        merged_spans = merge_segments(spans, max_gap_frames)
        for start_f, end_f in merged_spans:
            start_time = start_f / fps
            end_time = end_f / fps
            duration = end_time - start_time
            if duration >= 0.5: # Log only appearances longer than 0.5s
                writer.writerow([face_id, f"{start_time:.2f}", f"{end_time:.2f}", f"{duration:.2f}"])

print(f"[INFO] Log saved to {LOG_CSV}")
print("[INFO] Done.")