import streamlit as st
import os
import json
import csv
import numpy as np
import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from datetime import datetime
import tempfile
import pandas as pd

# Conditional import for FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    st.warning("FAISS library not found. Falling back to slower, brute-force similarity search.")

# --- Main Processing Function ---
def run_face_analysis(input_video_path, output_video_path, face_db_json, log_csv, config):
    # --- Configuration from Streamlit UI ---
    SIMILARITY_THRESHOLD = config["SIMILARITY_THRESHOLD"]
    EMBEDDING_BUFFER_SIZE = config["EMBEDDING_BUFFER_SIZE"]
    FACE_DETECT_INTERVAL = config["FACE_DETECT_INTERVAL"]
    PROCESS_FPS = config["PROCESS_FPS"]
    MIN_TRACK_SECONDS = config["MIN_TRACK_SECONDS"]

    # --- Model Initialization (cached) ---
    @st.cache_resource
    def load_models():
        yolo_model = YOLO("yolo11s.pt") # Ensure your model path is correct
        face_app_model = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        face_app_model.prepare(ctx_id=-1, det_size=(320, 320))
        return yolo_model, face_app_model

    yolo, face_app = load_models()

    def normalize_vec(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else None

    # --- Robustly Load or Init Face DB ---
    face_db = {}
    next_face_id = 0
    if os.path.exists(face_db_json):
        try:
            with open(face_db_json, 'r') as f:
                raw = json.load(f)
            if raw:
                face_db = {int(k): [np.array(e, dtype=np.float32) for e in v] for k, v in raw.items()}
                if face_db:
                    next_face_id = max(face_db.keys()) + 1
        except json.JSONDecodeError:
            st.warning(f"Could not read '{face_db_json}'. Starting with a new database.")
            face_db, next_face_id = {}, 0

    # --- Build FAISS index ---
    faiss_index = None
    pos_to_fid = []
    def build_faiss():
        nonlocal faiss_index, pos_to_fid
        if not FAISS_AVAILABLE: return
        pos_to_fid.clear()
        all_vecs = []
        for fid in sorted(face_db.keys()):
            for emb in face_db[fid]:
                norm_emb = normalize_vec(emb)
                if norm_emb is not None:
                    all_vecs.append(norm_emb)
                    pos_to_fid.append(fid)
        if not all_vecs:
            faiss_index = None; return
        embedding_dim = all_vecs[0].shape[0]
        faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss_index.add(np.vstack(all_vecs).astype('float32'))

    build_faiss()

    # --- Video I/O Setup ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video {input_video_path}"); return None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, PROCESS_FPS, (width, height))

    frame_count = 0
    track_to_face = {}
    pending_tracks = {}
    face_log = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(1, int(fps / PROCESS_FPS))
    progress_bar = st.progress(0, text="Processing video...")
    video_placeholder = st.empty()

    # --- Main Processing Loop (Unchanged) ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue
        
        # All of your core detection and matching logic inside this loop is untouched.
        # ... (Code from your script) ...
        results = yolo.track(frame, persist=True, classes=[0], verbose=False, imgsz=320)[0]
        person_boxes = []
        if results.boxes is not None and results.boxes.id is not None:
            for box in results.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                track_id = int(box.id.item())
                px1, py1, px2, py2 = coords
                person_boxes.append((track_id, px1, py1, px2, py2))
                if track_id not in pending_tracks and track_id not in track_to_face:
                    pending_tracks[track_id] = {'start_frame': frame_count}

        cached_faces = []
        if frame_count % FACE_DETECT_INTERVAL == 0:
            for track_id, px1, py1, px2, py2 in person_boxes:
                face_crop = frame[py1:py1 + int((py2-py1)*0.8), px1:px2]
                if face_crop.size == 0: continue
                try:
                    faces = face_app.get(face_crop)
                    if faces:
                        f = faces[0]
                        emb = getattr(f, "embedding", None)
                        bbox = getattr(f, "bbox", None)
                        if emb is not None and bbox is not None:
                            fx1, fy1, fx2, fy2 = (px1 + bbox[0], py1 + bbox[1], px1 + bbox[2], py1 + bbox[3])
                            cached_faces.append((np.asarray(emb, dtype=np.float32), (fx1, fy1, fx2, fy2), track_id))
                except Exception:
                    continue

        current_faces = []
        for embedding, bbox, track_id in cached_faces:
            norm_emb = normalize_vec(embedding)
            if norm_emb is None: continue

            matched_id = track_to_face.get(track_id)
            if matched_id is None:
                duration = (frame_count - pending_tracks.get(track_id, {}).get('start_frame', frame_count)) / fps
                if duration < MIN_TRACK_SECONDS: continue

                found_match_id = None
                if face_db:
                    if FAISS_AVAILABLE and faiss_index is not None and faiss_index.ntotal > 0:
                        D, I = faiss_index.search(norm_emb[np.newaxis, :].astype('float32'), 1)
                        if D[0][0] >= SIMILARITY_THRESHOLD: found_match_id = pos_to_fid[I[0][0]]
                    if found_match_id is None:
                        best_sim, best_fid = -1, None
                        for fid, embs in face_db.items():
                            for e in embs:
                                sim = np.dot(norm_emb, e)
                                if sim > best_sim: best_sim, best_fid = sim, fid
                        if best_sim >= SIMILARITY_THRESHOLD: found_match_id = best_fid
                
                if found_match_id is None:
                    matched_id = next_face_id
                    next_face_id += 1
                    face_db[matched_id] = []
                else:
                    matched_id = found_match_id
                
                track_to_face[track_id] = matched_id
                if track_id in pending_tracks: del pending_tracks[track_id]
            
            face_db.setdefault(matched_id, []).append(norm_emb)
            if len(face_db[matched_id]) > EMBEDDING_BUFFER_SIZE: face_db[matched_id].pop(0)
            build_faiss()
            current_faces.append((matched_id, bbox))
            face_log.setdefault(matched_id, []).append([frame_count, frame_count])
            if len(face_log[matched_id]) > 1:
                if frame_count - face_log[matched_id][-2][1] <= FACE_DETECT_INTERVAL * 2:
                    face_log[matched_id][-2][1] = face_log[matched_id][-1][1]
                    face_log[matched_id].pop()
        
        for track_id, px1, py1, px2, py2 in person_boxes: cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        for fid, (fx1, fy1, fx2, fy2) in current_faces:
            cv2.rectangle(frame, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (255, 0, 0), 2)
            cv2.putText(frame, f"FaceID: {fid}", (int(fx1), int(fy1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Processing Frame {frame_count}/{total_frames}")
        out.write(frame)
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    progress_bar.empty(); video_placeholder.empty()

    # The JSON save is also unchanged
    with open(face_db_json, 'w') as f:
        serial = {str(fid): [e.tolist() for e in embs] for fid, embs in face_db.items()}
        json.dump(serial, f, indent=2)

    def merge_segments(segments, max_gap_frames):
        if not segments: return []
        segments.sort(); merged = [segments[0][:]]
        for start, end in segments[1:]:
            if start - merged[-1][1] <= max_gap_frames: merged[-1][1] = max(merged[-1][1], end)
            else: merged.append([start, end])
        return merged

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    max_gap_frames = int(5 * fps)
    
    # --- DEBUGGED SECTION ---
    current_run_data = []
    file_exists = os.path.exists(log_csv)
    with open(log_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(log_csv) == 0:
            headers = ["FaceID", "StartTime", "EndTime", "DurationSeconds", "RunTimestamp"]
            writer.writerow(headers)
        
        for face_id, segments in face_log.items():
            merged = merge_segments(segments, max_gap_frames)
            for start, end in merged:
                # --- THE FIX ---
                # Calculate the duration of this specific segment
                duration_seconds = (end - start) / fps
                
                # Only log the segment if its duration meets the minimum requirement
                if duration_seconds >= MIN_TRACK_SECONDS:
                    row_dict = {
                        "FaceID": face_id,
                        "StartTime": f"{start / fps:.2f}", 
                        "EndTime": f"{end / fps:.2f}",
                        "DurationSeconds": f"{duration_seconds:.2f}", 
                        "RunTimestamp": run_timestamp
                    }
                    writer.writerow(row_dict.values())
                    current_run_data.append(row_dict)
    
    # Return the filtered data
    return pd.DataFrame(current_run_data) if current_run_data else pd.DataFrame(), face_db


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="CCTV Analysis")
st.title("📹 CCTV Face Detection and Recognition")
st.write("Upload a video, adjust parameters in the sidebar, and process the footage.")

FACE_DB_JSON = "face_db.json"
LOG_CSV = "face_log.csv"

# --- MODIFIED: Added session_state initialization ---
if 'face_db' not in st.session_state:
    if os.path.exists(FACE_DB_JSON):
        try:
            with open(FACE_DB_JSON, 'r') as f:
                raw_db = json.load(f)
            st.session_state.face_db = {int(k): [np.array(e) for e in v] for k,v in raw_db.items()}
        except (json.JSONDecodeError, ValueError):
            st.session_state.face_db = {}
    else:
        st.session_state.face_db = {}

st.sidebar.header("⚙️ Model Configuration")
conf_sim_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.4, 0.05)
conf_emb_buffer = st.sidebar.number_input("Embedding Buffer Size", 1, 100, 20, 1)
conf_face_interval = st.sidebar.number_input("Face Detect Interval (frames)", 1, 20, 2, 1)
conf_process_fps = st.sidebar.slider("Processing FPS", 1, 30, 20, 1)
conf_min_track_sec = st.sidebar.slider("Min Track Duration (seconds)", 0.0, 5.0, 1.0, 0.1)

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t_in:
        t_in.write(uploaded_file.read()); input_video_path = t_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t_out:
        output_video_path = t_out.name

    if st.button("Start Processing", type="primary"):
        config = {
            "SIMILARITY_THRESHOLD": conf_sim_threshold, "EMBEDDING_BUFFER_SIZE": conf_emb_buffer,
            "FACE_DETECT_INTERVAL": conf_face_interval, "PROCESS_FPS": conf_process_fps,
            "MIN_TRACK_SECONDS": conf_min_track_sec
        }
        
        # Pass the session state DB into the function
        results_df, updated_face_db = run_face_analysis(input_video_path, output_video_path, FACE_DB_JSON, LOG_CSV, config)
        
        # Update session state and save the file
        st.session_state.face_db = updated_face_db
        with open(FACE_DB_JSON, 'w') as f:
            json.dump({str(k): [e.tolist() for e in v] for k, v in updated_face_db.items()}, f, indent=2)

        st.success("✅ Processing Complete!")
        st.subheader("Processed Video")
        st.video(output_video_path)
        
        st.subheader("Appearance Log (Current Run)")
        if not results_df.empty:
            st.dataframe(results_df)
        else:
            st.warning("No appearances meeting the minimum duration were logged in this run.")

        st.subheader("Download Persistent Data")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(LOG_CSV):
                with open(LOG_CSV, "rb") as f:
                    st.download_button("⬇️ Download Full Log CSV", f, "face_log.csv")
        with col2:
            if os.path.exists(FACE_DB_JSON):
                with open(FACE_DB_JSON, "rb") as f:
                    st.download_button("⬇️ Download Face DB JSON", f, "face_db.json")
        
        os.remove(input_video_path); os.remove(output_video_path)