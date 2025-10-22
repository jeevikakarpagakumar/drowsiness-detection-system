# ==============================================================
# LOCAL DATA PREP (Alert / Drowsy)
# Sources: DDD, NTHU-DDD, Yawn dataset
# - face detect (Haar) → crop (+10%) → RGB → resize 160x160
# - merges datasets → train/val/test splits
# ==============================================================

import os, shutil, random, cv2
from pathlib import Path

random.seed(42)

# --------------------------- PATHS ---------------------------
BASE_PATH = r"C:\Users\kumar\OneDrive\Desktop\kavin proj"

# dataset roots
ddd_root   = os.path.join(BASE_PATH, "ddd", "Driver Drowsiness Dataset (DDD)")
nthu_root  = os.path.join(BASE_PATH, "nthudd", "train_data")
yawn_root  = os.path.join(BASE_PATH, "yawn")

# final processed dataset
BASE_OUTPUT = os.path.join(BASE_PATH, "processed_data_bin")
RAW_DIR   = os.path.join(BASE_OUTPUT, "raw")
FACE_DIR  = os.path.join(BASE_OUTPUT, "faces")
SPLIT_DIR = os.path.join(BASE_OUTPUT, "split")

# --------------------------- CONFIG ---------------------------
IMG_SIZE = 160
SPLIT_RATIO = (0.8, 0.1, 0.1)
IMG_EXTS = ('.jpg','.jpeg','.png','.bmp','.webp')

# -------------------- HELPERS --------------------
def copy_images(src, dst):
    """Copy all images from src → dst"""
    if not src or not os.path.exists(src): return 0
    names = [f for f in os.listdir(src) if f.lower().endswith(IMG_EXTS)]
    for f in names:
        try: shutil.copy(os.path.join(src, f), dst)
        except Exception: pass
    return len(names)

def detect_face_bbox(img_bgr):
    """Return largest face bbox"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.1, 4, minSize=(40,40))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        return (x, y, x+w, y+h)
    return None

def crop_resize_rgb(img_bgr, bbox=None, pad=0.10):
    """Crop by bbox + margin, resize to IMG_SIZE"""
    h, w = img_bgr.shape[:2]
    if bbox:
        x1, y1, x2, y2 = bbox
        mw, mh = int((x2-x1)*pad), int((y2-y1)*pad)
        x1, y1 = max(0, x1-mw), max(0, y1-mh)
        x2, y2 = min(w, x2+mw), min(h, y2+mh)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0: crop = img_bgr
    else:
        crop = img_bgr
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

def process_class(src_dir, dst_dir):
    """Detect, crop, resize faces for one class"""
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS)]
    total, saved = len(files), 0
    print(f"🔍 {Path(src_dir).name}: {total} imgs")
    for i, fn in enumerate(files, 1):
        p = os.path.join(src_dir, fn)
        img = cv2.imread(p)
        if img is None: continue
        bbox = detect_face_bbox(img)
        face_rgb = crop_resize_rgb(img, bbox)
        out = os.path.join(dst_dir, Path(fn).with_suffix(".jpg").name)
        cv2.imwrite(out, cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
        saved += 1
        if i % 500 == 0: print(f"  ... {i}/{total}")
    print(f"✅ Saved {saved} → {dst_dir}")

# -------------------- PREP FOLDERS --------------------
for d in [RAW_DIR, FACE_DIR, SPLIT_DIR]:
    if os.path.exists(d): shutil.rmtree(d)
os.makedirs(os.path.join(RAW_DIR,  "Alert"),  exist_ok=True)
os.makedirs(os.path.join(RAW_DIR,  "Drowsy"), exist_ok=True)
os.makedirs(os.path.join(FACE_DIR, "Alert"),  exist_ok=True)
os.makedirs(os.path.join(FACE_DIR, "Drowsy"), exist_ok=True)
for split in ["train","validation","test"]:
    os.makedirs(os.path.join(SPLIT_DIR, split, "Alert"),  exist_ok=True)
    os.makedirs(os.path.join(SPLIT_DIR, split, "Drowsy"), exist_ok=True)

# -------------------- STAGE 1: COPY RAW --------------------
print("\n📦 Copying raw images...")

# DDD
n_alert = copy_images(os.path.join(ddd_root, "Non Drowsy"), os.path.join(RAW_DIR, "Alert"))
n_drowsy = copy_images(os.path.join(ddd_root, "Drowsy"), os.path.join(RAW_DIR, "Drowsy"))

# NTHU
n_alert += copy_images(os.path.join(nthu_root, "notdrowsy"), os.path.join(RAW_DIR, "Alert"))
n_drowsy += copy_images(os.path.join(nthu_root, "drowsy"), os.path.join(RAW_DIR, "Drowsy"))

# Yawn
n_drowsy += copy_images(os.path.join(yawn_root, "yawn"), os.path.join(RAW_DIR, "Drowsy"))
n_alert  += copy_images(os.path.join(yawn_root, "no yawn"), os.path.join(RAW_DIR, "Alert"))

print(f"📊 Raw counts → Alert: {n_alert} | Drowsy: {n_drowsy}")

# -------------------- STAGE 2: FACE CROP --------------------
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("\n--- STAGE 2: FACE-CROP → RGB → RESIZE ---")
process_class(os.path.join(RAW_DIR, "Alert"),  os.path.join(FACE_DIR, "Alert"))
process_class(os.path.join(RAW_DIR, "Drowsy"), os.path.join(FACE_DIR, "Drowsy"))

# -------------------- STAGE 3: SPLIT --------------------
def split_copy(src_dir, split_root, cname, ratios=SPLIT_RATIO):
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(".jpg")]
    random.shuffle(files)
    n = len(files)
    n_tr, n_va = int(n*ratios[0]), int(n*ratios[1])
    tr, va, te = files[:n_tr], files[n_tr:n_tr+n_va], files[n_tr+n_va:]
    for f in tr: shutil.copy(os.path.join(src_dir, f), os.path.join(split_root, "train", cname))
    for f in va: shutil.copy(os.path.join(src_dir, f), os.path.join(split_root, "validation", cname))
    for f in te: shutil.copy(os.path.join(src_dir, f), os.path.join(split_root, "test", cname))
    return len(tr), len(va), len(te)

print("\n--- STAGE 3: SPLIT ---")
trA, vaA, teA = split_copy(os.path.join(FACE_DIR, "Alert"),  SPLIT_DIR, "Alert")
trD, vaD, teD = split_copy(os.path.join(FACE_DIR, "Drowsy"), SPLIT_DIR, "Drowsy")

print(f"📊 Alert  → train:{trA} val:{vaA} test:{teA}")
print(f"📊 Drowsy → train:{trD} val:{vaD} test:{teD}")

def total_in(folder):
    return sum(len([f for f in files if f.lower().endswith(".jpg")]) for _, _, files in os.walk(folder))


print("\n🎯 DONE! Final dataset root:", SPLIT_DIR)
for split in ["train","validation","test"]:
    a = total_in(os.path.join(SPLIT_DIR, split, "Alert"))
    d = total_in(os.path.join(SPLIT_DIR, split, "Drowsy"))
    print(f"{split.title():11s} → Alert: {a:6d} | Drowsy: {d:6d}")