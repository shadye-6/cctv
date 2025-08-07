import cv2
import numpy as np
from insightface.app import FaceAnalysis
from norfair import Detection, Tracker
from tracker import FaceTracker
from sklearn.metrics.pairwise import cosine_similarity

known_faces = []  
face_id_counter = 0

def match_face(embedding, known_faces, threshold=0.7):
    if not known_faces:
        return None
    similarities = cosine_similarity([embedding], [e for e, _ in known_faces])[0]
    max_sim = max(similarities)
    if max_sim > threshold:
        return known_faces[similarities.argmax()][1]
    return None

def crop_face(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    return image[y1:y2, x1:x2]

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize Norfair Tracker
tracker = FaceTracker()

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    tracked_objects = tracker.update(faces)

    for face, tracked_obj in zip(faces, tracked_objects):
        x1, y1, x2, y2 = face.bbox.astype(int)
        track_id = tracked_obj.id

        face_crop = crop_face(frame, face.bbox)
        embedding = face.embedding

        matched_id = match_face(embedding, known_faces)
        if matched_id is None:
            face_id_counter
            matched_id = face_id_counter
            known_faces.append((embedding, matched_id))
            face_id_counter += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {matched_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    

    print("faces : ", len(faces))
    print("ids: ", len(known_faces))

    cv2.imshow("Face Tracking + ReID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()