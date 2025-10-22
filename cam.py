# live_cam.py
import os, time, cv2, numpy as np, tensorflow as tf
from tensorflow import keras
from collections import deque

MODEL = r"C:\Users\kumar\OneDrive\Desktop\kavin proj\out_wsl\best_alltrained.keras"  # Windows path
# If running in WSL, use:
# MODEL = r"/mnt/c/Users/kumar/OneDrive/Desktop/kavin proj/out_wsl/drowsy_mnv2_retrained.keras"

IMG_SIZE = 160
CLASS_NAMES = ["Alert", "Drowsy"]
THRESH = 0.5
SMOOTH_WIN = 7               # moving average window
CAM_INDEX = 0                # default camera index

# Load model once
model = keras.models.load_model(MODEL, compile=False)

# Face detector (Haar)
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_largest_face(gray):
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,60))
    if len(faces) == 0:
        return None
    # pick largest by area
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    return (x, y, x+w, y+h)

def preprocess_face(bgr, bbox=None, pad=0.10):
    h, w = bgr.shape[:2]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        mh = int((y2-y1)*pad); mw = int((x2-x1)*pad)
        x1 = max(0, x1 - mw); y1 = max(0, y1 - mh)
        x2 = min(w, x2 + mw); y2 = min(h, y2 + mh)
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = bgr
    else:
        # fallback: center crop square
        side = min(h, w)
        cx, cy = w//2, h//2
        x1 = max(0, cx - side//2); y1 = max(0, cy - side//2)
        crop = bgr[y1:y1+side, x1:x1+side]
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    x = crop.astype(np.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # [-1,1]
    return np.expand_dims(x, 0), crop

def draw_overlay(frame, bbox, prob, avg_prob, fps):
    h, w = frame.shape[:2]
    if bbox is not None:
        x1,y1,x2,y2 = bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    p = float(prob)
    ap = float(avg_prob)
    label = "Drowsy" if ap >= THRESH else "Alert"
    color = (0,0,255) if label=="Drowsy" else (0,255,0)

    text = f"{label}  p={ap:.2f}  fps={fps:.1f}"
    cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # probability bar
    bar_w = int(300 * ap)
    cv2.rectangle(frame, (10, 50), (310, 75), (200,200,200), 1)
    cv2.rectangle(frame, (10, 50), (10+bar_w, 75), color, -1)
    return frame

def open_cam(index):
    cap = cv2.VideoCapture(index)
    # Try to set a nice size (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def main():
    cam_idx = CAM_INDEX
    cap = open_cam(cam_idx)
    if not cap.isOpened():
        print(f"Failed to open camera {cam_idx}")
        return

    probs = deque(maxlen=SMOOTH_WIN)
    t0 = time.time(); frames = 0

    print("Press 'q' to quit, 's' to save snapshot, 'c' to switch camera.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        frames += 1
        # FPS calc
        now = time.time()
        fps = frames / (now - t0 + 1e-6)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox = detect_largest_face(gray)

        x, face_vis = preprocess_face(frame, bbox, pad=0.10)
        prob = float(model.predict(x, verbose=0)[0][0])
        probs.append(prob)
        avg_prob = sum(probs) / len(probs)

        out = draw_overlay(frame.copy(), bbox, prob, avg_prob, fps)
        cv2.imshow("Drowsiness Monitor", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = os.path.join(os.getcwd(), f"snapshot_{int(time.time())}.jpg")
            cv2.imwrite(save_path, out)
            print("Saved", save_path)
        elif key == ord('c'):
            # switch camera
            cap.release()
            cam_idx = 1 - cam_idx  # toggle 0 <-> 1; change if you have more cams
            cap = open_cam(cam_idx)
            print(f"Switched to camera {cam_idx}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Make TF play nice with GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    main()