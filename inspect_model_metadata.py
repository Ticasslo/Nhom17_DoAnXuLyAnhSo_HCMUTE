import os
import pickle
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

script_dir = os.path.dirname(os.path.abspath(__file__))

# Chọn format model: 0 = h5, 1 = keras, 2 = saved_model
MODEL_TYPE = 2

# File metadata
METADATA_FILE = "saved_model/metadata.pkl"  # Hoặc "saved_model/metadata.pkl"

# Xác định model path
if MODEL_TYPE == 0:
    MODEL_PATH = os.path.join(script_dir, "gesture_model.h5")
elif MODEL_TYPE == 1:
    MODEL_PATH = os.path.join(script_dir, "best_model.keras")
elif MODEL_TYPE == 2:
    MODEL_PATH = os.path.join(script_dir, "saved_model")
else:
    raise ValueError(f"MODEL_TYPE phai la 0, 1 hoac 2, got {MODEL_TYPE}")

METADATA_PATH = os.path.join(script_dir, METADATA_FILE)

# METADATA
print("=" * 80)
print("METADATA.PKL")
print("=" * 80)

if not os.path.exists(METADATA_PATH):
    print(f"Khong tim thay: {METADATA_PATH}")
else:
    print(f"File: {METADATA_PATH}\n")
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Type: {type(metadata).__name__}")
    print(f"Keys: {list(metadata.keys())}")
    print()
    
    for key, value in metadata.items():
        print(f"{key}:")
        if isinstance(value, list):
            print(f"  Type: list, Length: {len(value)}")
            print(f"  Content: {value}")
        elif isinstance(value, dict):
            print(f"  Type: dict, Keys: {list(value.keys())}")
            print(f"  Content: {value}")
        else:
            print(f"  Type: {type(value).__name__}, Value: {value}")
        print()

# MODEL
print("=" * 80)
print("MODEL")
print("=" * 80)
import tensorflow as tf

if not os.path.exists(MODEL_PATH):
    print(f"Khong tim thay: {MODEL_PATH}")
else:
    print(f"File: {MODEL_PATH}")
    
    if MODEL_TYPE == 2:
        # SavedModel
        model = tf.saved_model.load(MODEL_PATH)
        print("Format: SavedModel")
        if hasattr(model, 'signatures'):
            print(f"Signatures: {list(model.signatures.keys())}")
            if 'serving_default' in model.signatures:
                sig = model.signatures['serving_default']
                print(f"Input: {sig.structured_input_signature}")
                print(f"Output: {sig.structured_outputs}")
    else:
        # Keras/H5
        model = tf.keras.models.load_model(MODEL_PATH)
        format_name = "Keras" if MODEL_TYPE == 1 else "H5"
        print(f"Format: {format_name}")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Number of layers: {len(model.layers)}")
        print(f"Total parameters: {model.count_params():,}")
