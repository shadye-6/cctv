import json
import numpy as np
import csv
from sklearn.cluster import DBSCAN

# ==========================
# Config
# ==========================
SIMILARITY_THRESHOLD = 0.635  # cosine similarity threshold
MIN_SAMPLES = 2               # min embeddings to form a cluster
OUTPUT_CSV = "face_clusters.csv"
LOG_CSV = "face_log.csv"     # <-- your log file: FaceID,StartFrame,EndFrame,Duration

def normalize_vec(v):
    """Normalize embedding to unit length (cosine similarity)."""
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-6)

def load_embeddings(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    ids = []
    embeddings = []

    for face_id, emb_list in data.items():
        arr = np.array(emb_list)  # shape (num_embeds, 512)
        mean_emb = np.mean(arr, axis=0)  # average across all embeddings
        ids.append(face_id)
        embeddings.append(mean_emb)

    return ids, np.array(embeddings)  # shape (num_faces, 512)

def load_logs(csv_file):
    """Load StartFrame, EndFrame, Duration for each FaceID from log CSV."""
    logs = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            face_id = row["FaceID"]
            start = row["StartFrame"]
            end = row["EndFrame"]
            duration = row["Duration"]
            logs.setdefault(face_id, []).append((start, end, duration))
    return logs

def cluster_embeddings(ids, embeddings):
    """Cluster embeddings into groups using DBSCAN + cosine similarity."""
    norm_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-6)
    cosine_sim = np.dot(norm_embeddings, norm_embeddings.T)
    cosine_dist = 1 - cosine_sim

    db = DBSCAN(eps=1-SIMILARITY_THRESHOLD, min_samples=MIN_SAMPLES, metric="precomputed")
    labels = db.fit_predict(cosine_dist)

    clusters = {}
    for face_id, label in zip(ids, labels):
        if label == -1:
            clusters.setdefault("noise", []).append(face_id)
        else:
            clusters.setdefault(label, []).append(face_id)
    return clusters

def save_clusters_to_csv(clusters, logs, csv_path=OUTPUT_CSV):
    """Save clusters into CSV format with StartFrame, EndFrame, Duration."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ClusterID", "FaceIDs", "StartFrames", "EndFrames", "Durations"])
        for label, group in clusters.items():
            group_starts, group_ends, group_durations = [], [], []
            for fid in group:
                if fid in logs:
                    for start, end, duration in logs[fid]:
                        group_starts.append(start)
                        group_ends.append(end)
                        group_durations.append(duration)
            writer.writerow([
                label,
                ",".join(group),
                ",".join(group_starts),
                ",".join(group_ends),
                ",".join(group_durations)
            ])
    print(f"\n✅ Clusters with frame ranges saved to {csv_path}")

def main(json_path, log_csv):
    ids, embeddings = load_embeddings(json_path)
    logs = load_logs(log_csv)

    # Filter embeddings to include only FaceIDs present in logs
    filtered_ids = []
    filtered_embeddings = []
    for fid, emb in zip(ids, embeddings):
        if fid in logs:
            filtered_ids.append(fid)
            filtered_embeddings.append(emb)
    filtered_embeddings = np.array(filtered_embeddings)

    # Cluster only filtered IDs
    clusters = cluster_embeddings(filtered_ids, filtered_embeddings)

    print("\n=== Face Clusters ===")
    for label, group in clusters.items():
        if label == "noise":
            print(f"Noise (no strong similarity): {group}")
        else:
            frames = []
            for fid in group:
                frames.extend(logs.get(fid, []))
            print(f"Cluster {label}: {group}, Frames={frames}")

    # Save to CSV
    save_clusters_to_csv(clusters, logs)

# ==========================
# Run
# ==========================
if __name__ == "__main__":
    main("face_db.json", LOG_CSV)   # <-- replace with your files
