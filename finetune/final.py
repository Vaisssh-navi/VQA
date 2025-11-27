import os
from collections import defaultdict

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
import pywt
import matplotlib.pyplot as plt

DATA_ROOT = r"C:\Users\Asus\OneDrive\Desktop\ransh_innovations\ell715A5\faces94"
FOLDERS_TO_USE = ["malestaff", "female"]
IMG_SIZE = (80, 60)       
N_PCA_COMPONENTS = 100
RANDOM_STATE = 0
PLOT_TSNE = True

def collect_subject_images():
    subject_to_paths = defaultdict(list)

    for folder in FOLDERS_TO_USE:
        folder_path = os.path.join(DATA_ROOT, folder)
        if not os.path.isdir(folder_path):
            print(f"[WARN] Folder missing: {folder_path}")
            continue

        # each subfolder inside is one person
        for subject in sorted(os.listdir(folder_path)):
            subject_dir = os.path.join(folder_path, subject)
            if not os.path.isdir(subject_dir):
                continue

            for fname in sorted(os.listdir(subject_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".pgm")):
                    img_path = os.path.join(subject_dir, fname)
                    subject_to_paths[subject].append(img_path)

    return subject_to_paths


def train_probe_split(subject_to_paths, train_ratio=0.75):
    rng = np.random.RandomState(RANDOM_STATE)
    gallery, probe = [], []

    for subject, paths in subject_to_paths.items():
        if len(paths) < 2:
            continue

        paths = sorted(paths)
        rng.shuffle(paths)

        n = len(paths)
        n_gallery = int(round(train_ratio * n))
        n_gallery = max(1, min(n_gallery, n - 1))

        gallery_paths = paths[:n_gallery]
        probe_paths = paths[n_gallery:]

        gallery.extend([(p, subject) for p in gallery_paths])
        probe.extend([(p, subject) for p in probe_paths])

    return gallery, probe


def load_image_gray(path):
    img = Image.open(path).convert("L")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def extract_raw_vector(path):
    arr = load_image_gray(path)
    return arr.flatten()


def extract_wavelet_features(path, wavelet="db2", level=2):
    arr = load_image_gray(path)
    coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=level)

    pieces = [coeffs[0].ravel()]  # approximation
    for (cH, cV, cD) in coeffs[1:]:
        pieces.append(cH.ravel())
        pieces.append(cV.ravel())
        pieces.append(cD.ravel())

    features = np.concatenate(pieces)
    return features


def build_feature_matrix(samples, extractor_fn):
    X_list, y_list = [], []

    for path, subj in samples:
        feat = extractor_fn(path)
        X_list.append(feat)
        y_list.append(subj)

    X = np.vstack(X_list)
    return X, y_list


def apply_pca(gallery_X, probe_X, n_components=N_PCA_COMPONENTS):
    # We can’t ask for more components than min(#gallery-1, feature_dim)
    max_components = min(gallery_X.shape[0] - 1, gallery_X.shape[1])
    if max_components <= 0:
        max_components = min(gallery_X.shape[0], gallery_X.shape[1])

    n_comp = min(n_components, max_components)

    pca = PCA(n_components=n_comp, whiten=True, random_state=RANDOM_STATE)
    gallery_Z = pca.fit_transform(gallery_X)
    probe_Z = pca.transform(probe_X)
    return gallery_Z, probe_Z, pca


def nearest_neighbor_accuracy(gallery_Z, gallery_labels, probe_Z, probe_labels):
    dists = euclidean_distances(probe_Z, gallery_Z)   # (n_probe, n_gallery)
    nn_indices = np.argmin(dists, axis=1)
    preds = [gallery_labels[i] for i in nn_indices]

    correct = sum(p == t for p, t in zip(preds, probe_labels))
    acc = correct / len(probe_labels) if probe_labels else 0.0
    return acc, preds


def plot_tsne(features, labels, title):
    print(f"[t-SNE] Running t-SNE on {len(labels)} gallery samples ...")
    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(features)

    unique_subjects = sorted(set(labels))
    subj_to_idx = {s: i for i, s in enumerate(unique_subjects)}
    colors = [subj_to_idx[s] for s in labels]

    plt.figure(figsize=(8, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=colors, s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.show()

def main():
    print("[STEP] Scanning dataset structure ...")
    subject_to_paths = collect_subject_images()
    n_subjects = len(subject_to_paths)
    n_images = sum(len(v) for v in subject_to_paths.values())
    print(f"      → {n_subjects} subjects, {n_images} total images\n")

    # Build gallery / probe split
    gallery, probe = train_probe_split(subject_to_paths, train_ratio=0.75)
    print(f"      → Gallery images : {len(gallery)}")
    print(f"      → Probe images   : {len(probe)}\n")

    if not gallery or not probe:
        print("[ERROR] Not enough images to create gallery/probe sets. Check DATA_ROOT.")
        return

    print("=============== PART 1 : EigenFaces =================")
    Xg_raw, y_gallery = build_feature_matrix(gallery, extract_raw_vector)
    Xp_raw, y_probe = build_feature_matrix(probe, extract_raw_vector)

    Zg_eig, Zp_eig, _ = apply_pca(Xg_raw, Xp_raw)

    eig_acc, eig_preds = nearest_neighbor_accuracy(Zg_eig, y_gallery, Zp_eig, y_probe)
    print(f"[PART 1 RESULT] EigenFaces accuracy = {eig_acc * 100:.2f}%\n")

    print("============= PART 2 : Wavelet Features =============")

    Xg_wav_raw, y_gallery_wav = build_feature_matrix(gallery, extract_wavelet_features)
    Xp_wav_raw, y_probe_wav = build_feature_matrix(probe, extract_wavelet_features)

    Zg_wav, Zp_wav, _ = apply_pca(Xg_wav_raw, Xp_wav_raw)

    wav_acc, wav_preds = nearest_neighbor_accuracy(Zg_wav, y_gallery_wav, Zp_wav, y_probe_wav)
    print(f"[PART 2 RESULT] Wavelet-based accuracy = {wav_acc * 100:.2f}%\n")

    print("============= PART 3 : Comparison & t-SNE ===========")

    print(f"[PART 3] EigenFaces  → {eig_acc * 100:.2f}%")
    print(f"[PART 3] Wavelets    → {wav_acc * 100:.2f}%\n")

    if wav_acc > eig_acc:
        print("[PART 3] Wavelet features work better on this split.\n")
    elif eig_acc > wav_acc:
        print("[PART 3] EigenFaces work better on this split.\n")
    else:
        print("[PART 3] Both methods reach the same accuracy on this split.\n")

    if PLOT_TSNE:
        print("[PART 3] Creating t-SNE plots for gallery features (may take a bit) ...")
        plot_tsne(Zg_eig, y_gallery, "t-SNE of EigenFaces (gallery)")
        plot_tsne(Zg_wav, y_gallery_wav, "t-SNE of Wavelet features (gallery)")


if __name__ == "__main__":
    main()
