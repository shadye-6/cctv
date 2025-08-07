import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# ======= CONFIG =======
input_path = r"C:\Users\dxksh\Downloads\videoplayback.mp4"
output_path = "output.mp4"
similarity_threshold = 0.5
embedding_buffer_size = 10
# =======================

# === Initialize Models ===
yolo = YOLO("yolov8n.pt")  # or 'yolov8s.pt' for better accuracy
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# === Load video ===
cap = cv2.VideoCapture(input_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === Face ID memory ===
face_db = {}  # {face_id: [embeddings]}
next_face_id = 0

# === Tracker ↔ Face ID mapping ===
track_to_face = {}  # {track_id: face_id}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ==== FACE DETECTION (Full Frame) ====
    faces = face_app.get(frame)

    current_faces = []  # to associate with tracks

    for face in faces:
        embedding = face.embedding
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # ==== FACE RE-ID MATCHING ====
        matched_id = None
        for face_id, embeddings in face_db.items():
            for stored_emb in embeddings:
                dist = cosine(embedding, stored_emb)
                if dist < similarity_threshold:
                    matched_id = face_id
                    break
            if matched_id is not None:
                break

        if matched_id is None:
            matched_id = next_face_id
            face_db[matched_id] = []
            next_face_id += 1

        # Add embedding to memory
        if len(face_db[matched_id]) >= embedding_buffer_size:
            face_db[matched_id].pop(0)
        face_db[matched_id].append(embedding)

        # Draw face box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"FaceID: {matched_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Save to list for matching to person boxes
        current_faces.append((matched_id, (x1, y1, x2, y2)))

    # ==== PERSON DETECTION + TRACKING ====
    results = yolo.track(frame, persist=True, classes=[0], verbose=False)[0]

    if results.boxes is not None:
        for box in results.boxes:
            person_box = box.xyxy[0].cpu().numpy().astype(int)
            px1, py1, px2, py2 = person_box

            # Get YOLO track_id
            track_id = int(box.id.item()) if box.id is not None else None
            if track_id is None:
                continue

            # === Match this tracked person to nearest face ===
            matched_face_id = track_to_face.get(track_id, None)
            for fid, (fx1, fy1, fx2, fy2) in current_faces:
                face_center = np.array([(fx1 + fx2) / 2, (fy1 + fy2) / 2])
                if px1 <= face_center[0] <= px2 and py1 <= face_center[1] <= py2:
                    matched_face_id = fid
                    track_to_face[track_id] = fid
                    break

            # Draw person box with face ID
            label = f"ID: {matched_face_id}" if matched_face_id is not None else "ID: ?"
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(frame, label, (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ==== Write frame ====
    out.write(frame)
    cv2.imshow("YOLO + InsightFace", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
