# %pip uninstall -y tensorflow tensorflow-intel keras
# %pip install tensorflow==2.16.1
# %pip uninstall -y jax jaxlib ml_dtypes
# %pip install tensorflow==2.16.1
# import tensorflow as tf
# print(tf.__version__)  # phải là 2.16.1
# from google.colab import files
# files.upload()

# TRAIN GESTURE MODEL - 46 FEATURES
# WITH LANDMARK AUGMENTATION (ROTATION + NOISE) - TRAIN ONLY
# SAVE: SavedModel (BEST + LAST) + best_model.keras
# TensorFlow 2.16.1 / Keras 3

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# CONFIG
CSV_PATH = "dataset.csv"

BEST_KERAS_PATH = "best_model.keras"      # best checkpoint (Keras format)
BEST_SAVEDMODEL_DIR = "saved_model_best"  # folder SavedModel best
LAST_SAVEDMODEL_DIR = "saved_model_last"  # folder SavedModel last
META_PATH = "metadata.pkl"

BATCH_SIZE = 128
EPOCHS = 120
LR = 1e-3
SEED = 42

# Augmentation config (TRAIN ONLY)
ROTATION_DEG = 8          # random rotation ±8 degrees
NOISE_STD = 0.003         # gaussian noise std
AUGMENT_TIMES = 1         # each train sample generates +N augmented copies

# 0) REPRODUCIBILITY
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# 1) LOAD DATASET
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Không tìm thấy {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

feat_cols = [c for c in df.columns if c.startswith("feat_")]
if len(feat_cols) != 46:
    raise ValueError(f"Dataset có {len(feat_cols)} features, cần đúng 46 (feat_*)")

if "label" not in df.columns:
    raise ValueError("Dataset thiếu cột 'label'")

df = df.dropna(subset=feat_cols + ["label"]).reset_index(drop=True)

X = df[feat_cols].astype(np.float32).values
y_str = df["label"].astype(str).values

print("Dataset loaded:", df.shape)
print("X shape:", X.shape)

# 2) LABEL ENCODER
le = LabelEncoder()
y = le.fit_transform(y_str)
num_classes = len(le.classes_)

print("Num classes:", num_classes)
print("Labels:", list(le.classes_))

# 3) TRAIN / VAL / TEST SPLIT
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print("Split:")
print("   Train:", X_train.shape)
print("   Val  :", X_val.shape)
print("   Test :", X_test.shape)

# 4) DATA AUGMENTATION (TRAIN ONLY)
def rotate_xy_pairs(X_in, max_deg, pair_end=42):
    X_out = X_in.copy()

    angle = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]], dtype=np.float32)

    # rotate each (x,y) pair
    for i in range(0, pair_end, 2):
        pts = X_in[:, i:i+2]          # (N,2)
        X_out[:, i:i+2] = pts @ R.T

    return X_out

def add_gaussian_noise(X_in, std, noise_on_all=True, pair_end=42):
    X_out = X_in.copy()
    if std <= 0:
        return X_out

    if noise_on_all:
        noise = np.random.normal(0, std, X_in.shape).astype(np.float32)
        X_out = X_out + noise
    else:
        noise = np.random.normal(0, std, (X_in.shape[0], pair_end)).astype(np.float32)
        X_out[:, :pair_end] = X_out[:, :pair_end] + noise

    return X_out

def augment_train_data(X_in, y_in, times=1, rot_deg=8, noise_std=0.003):
    X_list = [X_in]
    y_list = [y_in]

    for _ in range(times):
        X_rot = rotate_xy_pairs(X_in, max_deg=rot_deg, pair_end=42)
        X_aug = add_gaussian_noise(X_rot, std=noise_std, noise_on_all=True, pair_end=42)

        X_list.append(X_aug)
        y_list.append(y_in)

    return np.vstack(X_list).astype(np.float32), np.hstack(y_list).astype(np.int32)

X_train_aug, y_train_aug = augment_train_data(
    X_train, y_train,
    times=AUGMENT_TIMES,
    rot_deg=ROTATION_DEG,
    noise_std=NOISE_STD
)

print("After augmentation:")
print("   X_train_aug:", X_train_aug.shape)
print("   y_train_aug:", y_train_aug.shape)

# 5) CLASS WEIGHT (ON AUGMENTED TRAIN)
classes = np.unique(y_train_aug)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_aug
)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
print("Class weight computed.")

# 6) BUILD MODEL
model = models.Sequential([
    layers.Input(shape=(46,)),

    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.4),

    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.3),

    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 7) CALLBACKS
checkpoint_cb = callbacks.ModelCheckpoint(
    filepath=BEST_KERAS_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

cbs = [
    callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=1
    ),
    checkpoint_cb
]

# 8) TRAIN
history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_weight=class_weight,
    callbacks=cbs,
    verbose=1
)

# 9) TEST EVAL
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTEST ACCURACY: {test_acc:.4f}")
print(f"TEST LOSS    : {test_loss:.4f}")

# 10) EXPORT SAVEDMODEL (LAST + BEST)
def safe_rmtree(path):
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)

safe_rmtree(LAST_SAVEDMODEL_DIR)
safe_rmtree(BEST_SAVEDMODEL_DIR)

print("Export LAST SavedModel ->", LAST_SAVEDMODEL_DIR)
model.export(LAST_SAVEDMODEL_DIR)

print("Export BEST SavedModel ->", BEST_SAVEDMODEL_DIR)
best_model = tf.keras.models.load_model(BEST_KERAS_PATH)
best_model.export(BEST_SAVEDMODEL_DIR)

# 11) SAVE METADATA
metadata = {
    "num_features": 46,
    "feature_columns": feat_cols,
    "labels": list(le.classes_),
    "tf_version": tf.__version__,
    "keras_version": tf.keras.__version__,
    "formats": {
        "best_keras": BEST_KERAS_PATH,
        "saved_model_best": BEST_SAVEDMODEL_DIR,
        "saved_model_last": LAST_SAVEDMODEL_DIR
    },
    "augmentation": {
        "rotation_deg": ROTATION_DEG,
        "noise_std": NOISE_STD,
        "augment_times": AUGMENT_TIMES,
        "rotation_pairs_end": 42,
        "noise_on_all_features": True
    },
    "note": "Rotation applies to (feat_0..feat_41) x,y landmark pairs only."
}

with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("\nSaved:")
print("   - BEST .keras      :", BEST_KERAS_PATH)
print("   - BEST SavedModel  :", BEST_SAVEDMODEL_DIR)
print("   - LAST SavedModel  :", LAST_SAVEDMODEL_DIR)
print("   - Metadata         :", META_PATH)

# 12) PLOT ACC & LOSS (SAVE PNG)
hist = history.history

plt.figure(dpi=150)
plt.plot(hist.get("accuracy", []), label="Train Acc")
plt.plot(hist.get("val_accuracy", []), label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy.png", dpi=300)
plt.show()

plt.figure(dpi=150)
plt.plot(hist.get("loss", []), label="Train Loss")
plt.plot(hist.get("val_loss", []), label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss.png", dpi=300)
plt.show()
