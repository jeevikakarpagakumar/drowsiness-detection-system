# head_warmup_and_eval.py
import os, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----------------- CONFIG -----------------
IMG_SIZE = 160
BATCH = 64
AUTOTUNE = tf.data.AUTOTUNE

DATA_ROOT = r"/mnt/c/Users/kumar/OneDrive/Desktop/kavin proj/processed_data_bin/split"
OUT_DIR   = r"/mnt/c/Users/kumar/OneDrive/Desktop/kavin proj/out_wsl"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------- GPU QoL -----------------
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print("GPUs:", gpus)

# ----------------- DATASETS -----------------
def make_ds(split, shuffle):
    ds = keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_ROOT, split),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        label_mode="binary",
        shuffle=shuffle
    )
    return ds

train_raw = make_ds("train", shuffle=True)
val_raw   = make_ds("validation", shuffle=False)
test_raw  = make_ds("test", shuffle=False)
class_names = train_raw.class_names
print("Classes:", class_names)  # typically ['Alert','Drowsy']

def prep_for_mnv2(ds):
    def _pp(x, y):
        x = tf.cast(x, tf.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # [-1,1]
        return x, y
    return ds.map(_pp, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

train = prep_for_mnv2(train_raw)
val   = prep_for_mnv2(val_raw)
test  = prep_for_mnv2(test_raw)

# ----------------- BUILD MODEL -----------------
def build_model(img_size=IMG_SIZE, imagenet=True):
    inp = keras.Input((img_size, img_size, 3))
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=("imagenet" if imagenet else None),
        input_shape=(img_size, img_size, 3),
        alpha=0.75
    )
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="dense")(x)
    model = keras.Model(inp, out)
    return model, base

model, base = build_model()

# ----------------- HEAD WARM-UP -----------------
base.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])

cb = [
  keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, "best_head.keras"),
                                  monitor="val_auc", mode="max", save_best_only=True),
  keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                patience=3, restore_best_weights=True)
]

print("\n=== Train head (base frozen) ===")
model.fit(train, validation_data=val, epochs=5, callbacks=cb)

# ----------------- EVAL (after head warm-up) -----------------
def evaluate_and_print(m, test_ds, split_name="test"):
    probs = m.predict(test_ds, verbose=1).ravel()
    y_pred = (probs >= 0.5).astype(int)
    y_true = np.concatenate([y.numpy().ravel() for _, y in test_raw], axis=0).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[{split_name}] Confusion Matrix (rows=true, cols=pred):")
    print(f"labels: {class_names} (0={class_names[0]}, 1={class_names[1]})")
    print(cm)
    print(f"\n[{split_name}] Accuracy: {acc:.4f}\n")
    print(f"[{split_name}] Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

evaluate_and_print(model, test, "test (after head warm-up)")

# ----------------- OPTIONAL: FINE-TUNE LAST 30 LAYERS -----------------
# Uncomment this block if you want a bit more accuracy.
print("\n=== Fine-tune last 30 layers ===")
base.trainable = True
for L in base.layers[:-30]:
    L.trainable = False

model.compile(optimizer=keras.optimizers.Adam(3e-4),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])

cb2 = [
  keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, "best_finetune.keras"),
                                  monitor="val_auc", mode="max", save_best_only=True),
  keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                patience=3, restore_best_weights=True)
]

model.fit(train, validation_data=val, epochs=5, callbacks=cb2)

evaluate_and_print(model, test, "test (after fine-tune)")

# Save final usable model
model.save(os.path.join(OUT_DIR, "drowsy_mnv2_retrained.keras"))
print("\nSaved:", os.path.join(OUT_DIR, "drowsy_mnv2_retrained.keras"))