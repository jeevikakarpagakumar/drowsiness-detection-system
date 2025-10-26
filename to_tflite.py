# convert_to_tflite.py
import tensorflow as tf
from tensorflow import keras
import numpy as np

# ====== paths ======
IN_MODEL = r"C:\Users\kumar\OneDrive\Desktop\kavin proj\out_wsl\best_alltrained.keras"
OUT_FP32 = "best_alltrained_fp32.tflite"
OUT_INT8 = "best_alltrained_int8.tflite"   # optional (smaller/faster)

print("Loading Keras model...")
model = keras.models.load_model(IN_MODEL, compile=False)

# ---------- Plain FP32 TFLite ----------
print("Converting to FP32 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(OUT_FP32, "wb").write(tflite_model)
print(f"✅ Wrote {OUT_FP32}")

# ---------- Full int8 quantization (optional, recommended for Pi) ----------
# You need a small representative dataset of preprocessed samples.
# Replace this with ~100–500 sample images if possible.
IMG_SIZE = 160
def rep_data():
    for _ in range(100):
        # dummy data shaped like your input; replace with real preprocessed samples for best accuracy
        yield [np.random.randint(0,255,(IMG_SIZE,IMG_SIZE,3),dtype=np.uint8).astype(np.float32)]

print("Converting to INT8 TFLite (dynamic range) ...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# For full int8 (input/output int8), set supported_ops and provide proper rep data with correct scaling.
# If your model uses MobileNetV2 preprocess_input ([-1,1]), keep FP32 input/output but int8 weights (smaller, still fast).
converter.representative_dataset = rep_data
# Allow mixed float fallback to keep things easy:
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                       tf.lite.OpsSet.TFLITE_BUILTINS]
# If you want strict INT8 I/O, uncomment:
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()
open(OUT_INT8, "wb").write(tflite_int8)
print(f"✅ Wrote {OUT_INT8}")