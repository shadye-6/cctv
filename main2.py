import cv2
import numpy as np
import json
import os
import csv
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cdist

# ======= CONFIG =======
input_path = r"test.mp4"
output_path = "output_with_logs.mp4"
similarity_threshold = 0.6  # Stricter threshold for better accuracy
embedding_buffer_size = 10
face_detect_interval = 5
process_fps = 20

### NEW ### - Configuration for persistent database and logging
db_path = "face_database.json"
log_path = "tracking_log.csv"
# =======================


### NEW ### - Function to load the existing face database
def load_database(path):
    """Loads the face database from a JSON file."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            db_data = json.load(f)
            # Convert lists back to numpy arrays
            face_db = {int(k): [np.array(emb) for emb in v] for k, v in db_data["face_db"].items()}
            next_face_id = db_data["next_face_id"]
            print(f"Loaded {len(face_db)} faces from database.")
            return face_db, next_face_id
    return {}, 0

### NEW ### - Function to save the face database
def save_database(path, face_db, next_face_id):
    """Saves the face database to a JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    db_to_save = {k: [emb.tolist() for emb in v] for k, v in face_db.items()}
    with open(path, 'w') as f:
        json.dump({"face_db": db_to_save, "next_face_id": next_face_id}, f, indent=4)
    print(f"Saved database with {len(face_db)} faces.")

### NEW ### - Function to save the tracking log
def save_log(path, log_entries):
    """Saves the tracking log to a CSV file."""
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["FaceID", "StartTime", "EndTime", "DurationSeconds"]) # Header
        writer.writerows(log_entries)
    print(f"Saved {len(log_entries)} log entries.")


# === Initialize Models ===
yolo = YOLO("yolov8n.pt")  # Use 'n' for speed, 's' or 'm' for accuracy
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

### NEW ### - Load persistent face database
face_db, next_face_id = load_database(db_path)

# === Tracker ↔ Face ID mapping ===
track_to_face = {}  # {track_id: face_id}

### NEW ### - Data structures for logging
active_tracks = {} # {track_id: {"face_id": id, "start_frame": frame}}
log_entries = []   # List to hold finalized log data for saving

frame_count = 0
cached_faces = []

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # ==== PERSON DETECTION + TRACKING ====
        results = yolo.track(frame, persist=True, classes=[0], verbose=False, imgsz=320)[0]

        person_boxes = []
        current_track_ids = set()
        if results.boxes is not None:
            for box in results.boxes:
                person_box = box.xyxy[0].cpu().numpy().astype(int)
                track_id = int(box.id.item()) if box.id is not None else None
                if track_id is not None:
                    px1, py1, px2, py2 = person_box
                    person_boxes.append((track_id, px1, py1, px2, py2))
                    current_track_ids.add(track_id)

        # ==== FACE DETECTION every N frames ====
        if frame_count % face_detect_interval == 0:
            cached_faces = []
            for track_id, px1, py1, px2, py2 in person_boxes:
                # Crop a smaller region around the head for efficiency
                head_crop = frame[py1:int(py1 + (py2 - py1) * 0.4), px1:px2]
                if head_crop.size == 0: continue
                
                faces = face_app.get(head_crop)
                for face in faces:
                    embedding = face.embedding
                    # Adjust face bbox to global frame coordinates
                    bbox = face.bbox.astype(int)
                    fx1, fy1, fx2, fy2 = px1 + bbox[0], py1 + bbox[1], px1 + bbox[2], py1 + bbox[3]
                    cached_faces.append((embedding, (fx1, fy1, fx2, fy2), track_id))

        # ==== FACE RE-ID MATCHING ====
        current_faces_drawn = []
        for embedding, face_bbox, track_id in cached_faces:
            matched_id = None
            if face_db:
                # Flatten the database for efficient search
                all_ids = list(face_db.keys())
                all_embeddings = np.array([emb for emb_list in face_db.values() for emb in emb_list])
                
                # Calculate cosine distances and find the best match
                dists = cdist([embedding], all_embeddings, metric='cosine')[0]
                best_match_idx = np.argmin(dists)

                if dists[best_match_idx] < similarity_threshold:
                    # Find which face_id this best embedding belongs to
                    count = 0
                    for fid, emb_list in face_db.items():
                        if best_match_idx < count + len(emb_list):
                            matched_id = fid
                            break
                        count += len(emb_list)

            if matched_id is None:
                matched_id = next_face_id
                face_db[matched_id] = []
                next_face_id += 1

            # Update the buffer for the matched face
            if len(face_db[matched_id]) >= embedding_buffer_size:
                face_db[matched_id].pop(0)
            face_db[matched_id].append(embedding)

            track_to_face[track_id] = matched_id
            current_faces_drawn.append((matched_id, face_bbox))

        ### NEW ### - Timestamp Logging Logic
        # Check for tracks that have disappeared
        lost_track_ids = set(active_tracks.keys()) - current_track_ids
        for track_id in lost_track_ids:
            session = active_tracks.pop(track_id)
            start_time = session["start_frame"] / fps
            end_time = frame_count / fps
            duration = end_time - start_time
            if duration > 1: # Log only if present for more than 1 second
                log_entries.append([session["face_id"], f"{start_time:.2f}", f"{end_time:.2f}", f"{duration:.2f}"])

        # Check for new tracks
        for track_id, px1, py1, px2, py2 in person_boxes:
            if track_id not in active_tracks and track_id in track_to_face:
                active_tracks[track_id] = {
                    "face_id": track_to_face[track_id],
                    "start_frame": frame_count
                }

        # ==== DRAW RESULTS ====
        for track_id, px1, py1, px2, py2 in person_boxes:
            matched_face_id = track_to_face.get(track_id)
            label = f"ID: {matched_face_id}" if matched_face_id is not None else "ID: ?"
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for fid, (fx1, fy1, fx2, fy2) in current_faces_drawn:
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
            cv2.putText(frame, f"FaceID: {fid}", (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # ==== Write frame ====
        out.write(frame)
        cv2.imshow("YOLO + InsightFace", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

finally:
    # Cleanup and save data
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    ### NEW ### - Finalize any remaining active tracks
    for track_id, session in active_tracks.items():
        start_time = session["start_frame"] / fps
        end_time = frame_count / fps
        duration = end_time - start_time
        if duration > 1:
            log_entries.append([session["face_id"], f"{start_time:.2f}", f"{end_time:.2f}", f"{duration:.2f}"])
    
    # Save everything
    save_database(db_path, face_db, next_face_id)
    if log_entries:
        save_log(log_path, log_entries)

