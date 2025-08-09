import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cdist
import csv

# ======= CONFIG =======
input_path = r"test.mp4"
output_path = "output.mp4"
log_csv_path = "face_log.csv"
similarity_threshold = 0.5
embedding_buffer_size = 10
face_detect_interval = 2
process_fps = 20
reuse_face_id_window = 30
# =======================

# === Initialize Models ===
yolo = YOLO("yolo11n.pt")
face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))

# === Load video ===
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
skip_frames = max(1, int(fps / process_fps))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, process_fps, (width, height))

# === Face ID memory ===
face_db = {}
next_face_id = 0

track_to_face = {}
track_last_seen = {}

frame_count = 0
cached_faces = []

# === Face presence log ===
face_log = {}  # {face_id: [(start_frame, end_frame), ...]}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % skip_frames != 0:
        frame_count += 1
        continue

    current_faces = []

    # ==== PERSON DETECTION + TRACKING ====
    results = yolo.track(frame, persist=True, classes=[0], verbose=False, imgsz=320)[0]
    person_boxes = []
    if results.boxes is not None:
        for box in results.boxes:
            person_box = box.xyxy[0].cpu().numpy().astype(int)
            px1, py1, px2, py2 = person_box
            track_id = int(box.id.item()) if box.id is not None else None
            if track_id is not None:
                person_boxes.append((track_id, px1, py1, px2, py2))

    # ==== FACE DETECTION every N frames ====
    if frame_count % face_detect_interval == 0:
        cached_faces = []
        for track_id, px1, py1, px2, py2 in person_boxes:
            # face_h = int((py2 - py1) * 0.7)
            face_h = int((py2 - py1))
            head_crop = frame[py1:py1 + face_h, px1:px2]
            if head_crop.size == 0:
                continue
            faces = face_app.get(head_crop)
            for face in faces:
                embedding = face.embedding
                bbox = face.bbox.astype(int)
                fx1, fy1, fx2, fy2 = px1 + bbox[0], py1 + bbox[1], px1 + bbox[2], py1 + bbox[3]
                cached_faces.append((embedding, (fx1, fy1, fx2, fy2), track_id))

    # ==== FACE RE-ID MATCHING ====
    for embedding, (fx1, fy1, fx2, fy2), track_id in cached_faces:
        matched_id = None

        if track_id in track_to_face and frame_count - track_last_seen.get(track_id, 0) <= reuse_face_id_window:
            matched_id = track_to_face[track_id]
        else:
            if face_db:
                all_embeddings = np.array([emb for emb_list in face_db.values() for emb in emb_list])
                all_ids = np.array([fid for fid, emb_list in face_db.items() for _ in emb_list])
                dists = cdist([embedding], all_embeddings, metric='cosine')[0]
                best_idx = np.argmin(dists)
                if dists[best_idx] < similarity_threshold:
                    matched_id = all_ids[best_idx]

        if matched_id is None:
            matched_id = next_face_id
            face_db[matched_id] = []
            next_face_id += 1

        if len(face_db[matched_id]) >= embedding_buffer_size:
            face_db[matched_id].pop(0)
        face_db[matched_id].append(embedding)

        track_to_face[track_id] = matched_id
        track_last_seen[track_id] = frame_count
        current_faces.append((matched_id, (fx1, fy1, fx2, fy2)))

        # === LOG APPEARANCE ===
        if matched_id not in face_log:
            face_log[matched_id] = []

        if not face_log[matched_id] or frame_count - face_log[matched_id][-1][1] > face_detect_interval * 2:
            # New segment
            face_log[matched_id].append([frame_count, frame_count])
        else:
            # Update end frame of ongoing segment
            face_log[matched_id][-1][1] = frame_count

    # ==== CLEANUP OLD TRACKS ====
    for tid in list(track_last_seen.keys()):
        if frame_count - track_last_seen[tid] > reuse_face_id_window:
            track_to_face.pop(tid, None)
            track_last_seen.pop(tid, None)

    # ==== DRAW RESULTS ====
    for track_id, px1, py1, px2, py2 in person_boxes:
        matched_face_id = track_to_face.get(track_id, None)
        label = ""
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for fid, (fx1, fy1, fx2, fy2) in [(fid, bbox) for fid, bbox in current_faces]:
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
        cv2.putText(frame, f"FaceID: {fid}", (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow("YOLO + InsightFace", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

def merge_segments(segments, max_gap_frames):
    if not segments:
        return []

    segments.sort()
    merged = [segments[0]]

    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap_frames:
            merged[-1][1] = max(last_end, end)  # Extend previous
        else:
            merged.append([start, end])
    return merged

# ==== WRITE LOG TO CSV ====
with open(log_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["FaceID", "StartTime", "EndTime", "DurationSeconds"])
    
    max_gap_seconds = 5
    max_gap_frames = int(max_gap_seconds * fps)

    for face_id, spans in face_log.items():
        merged_spans = merge_segments(spans, max_gap_frames)

        # Calculate total duration for this face
        total_duration = sum((end_f - start_f) / fps for start_f, end_f in merged_spans)

        # Only log if total duration > 1 second
        if total_duration > 1:
            for start_f, end_f in merged_spans:
                start_time = round(start_f / fps, 2)
                end_time = round(end_f / fps, 2)
                duration = round(end_time - start_time, 2)
                writer.writerow([face_id, f"{start_time:.2f}", f"{end_time:.2f}", f"{duration:.2f}"])
