import os
import time
import cv2
import warnings
import numpy as np
import pickle
import sys
import csv
from collections import defaultdict

import mediapipe as mp
from mediapipe.tasks.python import vision
import tensorflow as tf

# Giảm warning log
warnings.filterwarnings("ignore", category=UserWarning)

#  0. CONSTANTS 
# Label prefixes
ASYMMETRIC_PREFIX_RH = "A_RH_"
ASYMMETRIC_PREFIX_LH = "A_LH_"
SYMMETRIC_PREFIX = "S_"

# MediaPipe landmarks
NUM_LANDMARKS = 21  # Số lượng landmarks cho mỗi hand
FINGER_MCP_INDICES = [5, 9, 13, 17]  # Index, Middle, Ring, Pinky MCPs
INDEX_MCP_IDX = 5   # Index finger MCP landmark index
PINKY_MCP_IDX = 17  # Pinky finger MCP landmark index
NUM_FEATURES = 46  # 42 landmarks (21 * 2) + 2 Y_hand + 2 X_hand = 46 features

# Filtering thresholds
GESTURE_REJECT_THRESHOLD = 0.5
ENTROPY_THRESHOLD = 2.5
ORIENTATION_MIN_MAGNITUDE = 0.10

#  1. Load Gesture Model & Metadata 
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

# Model paths
GESTURE_MODEL_PATH = os.path.join(project_root, "models", "SavedModel", "saved_model_best")
METADATA_PATH = os.path.join(project_root, "models", "SavedModel", "metadata.pkl")

# Load model
if not os.path.exists(GESTURE_MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model: {GESTURE_MODEL_PATH}")
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Không tìm thấy metadata: {METADATA_PATH}")

print("Đang tải model và metadata...")
print("  → Loading gesture model...")
gesture_model = tf.saved_model.load(GESTURE_MODEL_PATH)
print("  → Model loaded (SavedModel format)")

# Kiểm tra và lấy signature
if hasattr(gesture_model, 'signatures') and 'serving_default' in gesture_model.signatures:
    model_signature = gesture_model.signatures['serving_default']
    print(f"  → Using signature: serving_default")
else:
    raise ValueError("SavedModel không có signature 'serving_default'!")

print("  → Loading metadata...")
with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)
    
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata.pkl không phải dict! Type: {type(metadata)}")
    
    if 'labels' not in metadata:
        raise ValueError(f"metadata.pkl không có 'labels'!")
    
    labels_list = metadata['labels']
    if not isinstance(labels_list, list):
        raise ValueError(f"metadata['labels'] phải là list, nhận được: {type(labels_list)}")
    
    if len(labels_list) == 0:
        raise ValueError("metadata['labels'] không được rỗng!")
    
    # Tạo label_mapping: {0: label0, 1: label1, ...}
    label_mapping = {i: label for i, label in enumerate(labels_list)}
    
    # Extract symmetric gestures từ labels
    SYMMETRIC_GESTURES = set([label for label in labels_list if label.startswith(SYMMETRIC_PREFIX)])
    
    num_classes = len(labels_list)
    
    print(f"  → Loaded {num_classes} labels")
    print(f"  → Symmetric gestures: {len(SYMMETRIC_GESTURES)}")
    print(f"  → Model ready: {num_classes} classes")

#  2. Load MediaPipe Hand Landmarker
HAND_LANDMARKER_MODEL_PATH = os.path.join(project_root, "models", "hand_landmarker.task")

if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy model MediaPipe: {HAND_LANDMARKER_MODEL_PATH}"
    )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,  # IMAGE mode cho static images
    num_hands=1,  # Chỉ cần 1 tay cho batch evaluation
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)


#  3. Normalize Features Function (giống như trong real-time code)
def normalize_features(landmarks_array):
    landmarks = np.asarray(landmarks_array, dtype=np.float32)
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    
    # Relative normalization về wrist (landmark 0)
    wrist_x, wrist_y = x_coords[0], y_coords[0]
    relative_x = x_coords - wrist_x
    relative_y = y_coords - wrist_y
    
    # Scale normalization
    max_value = max(np.max(np.abs(relative_x)), np.max(np.abs(relative_y)))
    if max_value > 0:
        normalized_x = relative_x / max_value
        normalized_y = relative_y / max_value
    else:
        normalized_x, normalized_y = relative_x, relative_y
    
    # ORIENTATION FEATURES (HAND LOCAL SPACE)
    # Tính trung bình của 4 MCP joints (Index, Middle, Ring, Pinky)
    mcp_x_avg = np.mean(normalized_x[FINGER_MCP_INDICES])
    mcp_y_avg = np.mean(normalized_y[FINGER_MCP_INDICES])
    
    # Y_hand: vector từ wrist (0,0) đến mean(MCPs)
    y_hand_x = mcp_x_avg
    y_hand_y = mcp_y_avg
    
    # X_hand: vector từ index_MCP đến pinky_MCP
    x_hand_x = normalized_x[PINKY_MCP_IDX] - normalized_x[INDEX_MCP_IDX]
    x_hand_y = normalized_y[PINKY_MCP_IDX] - normalized_y[INDEX_MCP_IDX]
    
    EPSILON = 1e-6
    
    # Normalize Y_hand
    y_hand_mag = np.sqrt(y_hand_x**2 + y_hand_y**2)
    if y_hand_mag > EPSILON:
        y_hand_x_normalized = y_hand_x / y_hand_mag
        y_hand_y_normalized = y_hand_y / y_hand_mag
    else:
        y_hand_x_normalized = 0.0
        y_hand_y_normalized = -1.0
    
    # Normalize X_hand
    x_hand_mag = np.sqrt(x_hand_x**2 + x_hand_y**2)
    if x_hand_mag > EPSILON:
        x_hand_x_normalized = x_hand_x / x_hand_mag
        x_hand_y_normalized = x_hand_y / x_hand_mag
    else:
        # Fallback: vector vuông góc với Y_hand
        x_hand_x_normalized = -y_hand_y_normalized
        x_hand_y_normalized = y_hand_x_normalized
        x_hand_mag_fallback = np.sqrt(x_hand_x_normalized**2 + x_hand_y_normalized**2)
        if x_hand_mag_fallback > EPSILON:
            x_hand_x_normalized = x_hand_x_normalized / x_hand_mag_fallback
            x_hand_y_normalized = x_hand_y_normalized / x_hand_mag_fallback
        else:
            x_hand_x_normalized = 1.0
            x_hand_y_normalized = 0.0
    
    # Đảm bảo orientation vectors là unit vectors
    y_hand_final_mag = np.sqrt(y_hand_x_normalized**2 + y_hand_y_normalized**2)
    if abs(y_hand_final_mag - 1.0) > EPSILON and y_hand_final_mag > EPSILON:
        y_hand_x_normalized = y_hand_x_normalized / y_hand_final_mag
        y_hand_y_normalized = y_hand_y_normalized / y_hand_final_mag
    
    x_hand_final_mag = np.sqrt(x_hand_x_normalized**2 + x_hand_y_normalized**2)
    if abs(x_hand_final_mag - 1.0) > EPSILON and x_hand_final_mag > EPSILON:
        x_hand_x_normalized = x_hand_x_normalized / x_hand_final_mag
        x_hand_y_normalized = x_hand_y_normalized / x_hand_final_mag
    
    # Flatten thành NUM_FEATURES features
    feats = np.empty(NUM_FEATURES, dtype=np.float32)
    for i in range(NUM_LANDMARKS):
        feats[2*i] = float(normalized_x[i])
        feats[2*i+1] = float(normalized_y[i])
    
    # Thêm Y_hand features (indices 42, 43)
    feats[NUM_LANDMARKS * 2] = float(y_hand_x_normalized)
    feats[NUM_LANDMARKS * 2 + 1] = float(y_hand_y_normalized)
    
    # Thêm X_hand features (indices 44, 45)
    feats[NUM_LANDMARKS * 2 + 2] = float(x_hand_x_normalized)
    feats[NUM_LANDMARKS * 2 + 3] = float(x_hand_y_normalized)
    
    return feats

def check_orientation_validity(landmarks_array):
    landmarks = np.asarray(landmarks_array, dtype=np.float32)
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    
    wrist_x, wrist_y = x_coords[0], y_coords[0]
    relative_x = x_coords - wrist_x
    relative_y = y_coords - wrist_y
    
    max_value = max(np.max(np.abs(relative_x)), np.max(np.abs(relative_y)))
    if max_value > 0:
        normalized_x = relative_x / max_value
        normalized_y = relative_y / max_value
    else:
        normalized_x, normalized_y = relative_x, relative_y
    
    mcp_x_avg = np.mean(normalized_x[FINGER_MCP_INDICES])
    mcp_y_avg = np.mean(normalized_y[FINGER_MCP_INDICES])
    
    y_hand_x = mcp_x_avg
    y_hand_y = mcp_y_avg
    y_hand_mag = np.sqrt(y_hand_x**2 + y_hand_y**2)
    
    x_hand_x = normalized_x[PINKY_MCP_IDX] - normalized_x[INDEX_MCP_IDX]
    x_hand_y = normalized_y[PINKY_MCP_IDX] - normalized_y[INDEX_MCP_IDX]
    x_hand_mag = np.sqrt(x_hand_x**2 + x_hand_y**2)
    
    is_valid = (y_hand_mag >= ORIENTATION_MIN_MAGNITUDE and 
                x_hand_mag >= ORIENTATION_MIN_MAGNITUDE)
    
    return is_valid, y_hand_mag, x_hand_mag

def map_prediction_label(prediction, SYMMETRIC_GESTURES):
    # Symmetric gestures: bỏ prefix "S_"
    if prediction in SYMMETRIC_GESTURES:
        return prediction.replace(SYMMETRIC_PREFIX, "") if prediction.startswith(SYMMETRIC_PREFIX) else prediction
    
    # Asymmetric gestures: bỏ prefix "A_RH_" hoặc "A_LH_"
    if prediction.startswith(ASYMMETRIC_PREFIX_RH):
        return prediction.replace(ASYMMETRIC_PREFIX_RH, "RH_")
    elif prediction.startswith(ASYMMETRIC_PREFIX_LH):
        return prediction.replace(ASYMMETRIC_PREFIX_LH, "LH_")
    
    return prediction

def normalize_label_for_comparison(label):
    # Bỏ prefix S_, A_RH_, A_LH_ để so sánh
    if label.startswith(SYMMETRIC_PREFIX):
        return label.replace(SYMMETRIC_PREFIX, "")
    elif label.startswith(ASYMMETRIC_PREFIX_RH):
        return label.replace(ASYMMETRIC_PREFIX_RH, "RH_")
    elif label.startswith(ASYMMETRIC_PREFIX_LH):
        return label.replace(ASYMMETRIC_PREFIX_LH, "LH_")
    return label

#  4. Predict Gesture Function
@tf.function(reduce_retracing=True)
def predict_gesture(features_batch):
    if isinstance(features_batch, np.ndarray):
        features_batch = tf.constant(features_batch, dtype=tf.float32)
    elif not isinstance(features_batch, tf.Tensor):
        features_batch = tf.convert_to_tensor(features_batch, dtype=tf.float32)
    
    result = model_signature(input_layer_1=features_batch)
    return result['output_0']

#  5. Process Image Function
def process_image(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        return None, "Không đọc được ảnh"
    
    # Convert BGR sang RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False
    
    # Tạo MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Detect hand landmarks
    result = landmarker.detect(mp_image)
    
    # Kiểm tra có hand không
    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        return None, "Không detect được hand"
    
    # Lấy hand đầu tiên
    landmarks = result.hand_landmarks[0]
    
    # Validate số lượng landmarks
    if len(landmarks) != NUM_LANDMARKS:
        return None, f"Số lượng landmarks không đúng: {len(landmarks)}"
    
    # Convert landmarks sang numpy array
    landmarks_normalized = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    landmarks_normalized = np.clip(landmarks_normalized, 0.0, 1.0)
    
    # Kiểm tra orientation validity
    orientation_valid, y_hand_mag, x_hand_mag = check_orientation_validity(landmarks_normalized)
    if not orientation_valid:
        return None, f"Orientation không ổn định (Y:{y_hand_mag:.3f}, X:{x_hand_mag:.3f})"
    
    # Extract features
    features = normalize_features(landmarks_normalized).astype(np.float32)
    
    # Predict gesture
    features_reshaped = features.reshape(1, NUM_FEATURES).astype(np.float32)
    predictions = predict_gesture(features_reshaped)
    predictions_np = predictions.numpy() if hasattr(predictions, 'numpy') else np.array(predictions)
    probs = predictions_np[0]
    
    # Lấy top-1 prediction
    top1_idx = int(np.argmax(probs))
    confidence = float(probs[top1_idx])
    
    # Kiểm tra class_id hợp lệ
    if top1_idx not in label_mapping:
        return None, f"class_id {top1_idx} không có trong label_mapping"
    
    raw_label = label_mapping[top1_idx]
    
    # Tính entropy
    EPSILON = 1e-10
    entropy = -np.sum(probs * np.log(probs + EPSILON))
    
    # Filter: entropy
    if entropy > ENTROPY_THRESHOLD:
        return None, f"Entropy quá cao ({entropy:.2f})"
    
    # Filter: confidence
    if confidence < GESTURE_REJECT_THRESHOLD:
        return None, f"Confidence quá thấp ({confidence:.2f})"
    
    # Map prediction label
    predicted_label = map_prediction_label(raw_label, SYMMETRIC_GESTURES)
    
    return predicted_label, confidence

#  6. Main Evaluation Function
def evaluate_images(labels_dir):
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {labels_dir}")
    
    print(f"Bắt đầu đánh giá từ thư mục: {labels_dir}")
    
    # Lấy danh sách các folder (tên nhãn)
    labels = [f for f in os.listdir(labels_dir) 
              if os.path.isdir(os.path.join(labels_dir, f))]
    
    if len(labels) == 0:
        raise ValueError(f"Không tìm thấy folder nào trong {labels_dir}")
    
    print(f"Tìm thấy {len(labels)} nhãn:")
    for label in sorted(labels):
        print(f"  - {label}")
    print()
    
    # Kết quả
    results = []  # List of (image_path, true_label, predicted_label, is_correct, confidence, error_msg)
    stats_per_label = defaultdict(lambda: {'total': 0, 'processed': 0, 'correct': 0, 'no_hand': 0, 'error': 0})
    
    # Xử lý từng folder
    total_images = 0
    processed_images = 0
    
    for label in sorted(labels):
        label_path = os.path.join(labels_dir, label)  # Đường dẫn folder chứa ảnh
        
        # Lấy danh sách ảnh trong folder
        image_files = [f for f in os.listdir(label_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(image_files) == 0:
            print(f"  Folder '{label}' không có ảnh nào")
            continue
        
        print(f"Xử lý folder '{label}' ({len(image_files)} ảnh)...")
        
        # Normalize true label để so sánh
        true_label_normalized = normalize_label_for_comparison(label)
        
        for image_file in sorted(image_files):
            image_path = os.path.join(label_path, image_file)
            total_images += 1
            stats_per_label[label]['total'] += 1
            
            # Xử lý đặc biệt cho Nothing
            is_nothing_label = (label == 'S_Nothing' or label == 'Nothing' or 
                               label.startswith('S_Nothing') or label.startswith('Nothing'))
            
            # Process image
            predicted_label, result_info = process_image(image_path)
            
            if predicted_label is None:
                # Lỗi hoặc không detect được hand
                error_msg = result_info
                
                if is_nothing_label and "Không detect được hand" in error_msg:
                    # Folder là Nothing và không detect được hand → ĐÚNG
                    results.append((image_path, label, "No Hand", True, 1.0, ""))
                    stats_per_label[label]['correct'] += 1
                    stats_per_label[label]['processed'] += 1
                    processed_images += 1
                else:
                    # Folder không phải Nothing hoặc lỗi khác → SAI
                    results.append((image_path, label, None, False, 0.0, error_msg))
                    if "Không detect được hand" in error_msg:
                        stats_per_label[label]['no_hand'] += 1
                    else:
                        stats_per_label[label]['error'] += 1
            else:
                # Có prediction
                confidence = result_info
                processed_images += 1
                stats_per_label[label]['processed'] += 1
                
                # Xử lý đặc biệt: Nếu folder không phải Nothing nhưng predict Nothing → false positive
                if not is_nothing_label and (predicted_label == 'S_Nothing' or predicted_label == 'Nothing'):
                    # False positive: có hand nhưng predict Nothing
                    results.append((image_path, label, predicted_label, False, confidence, "False positive: Nothing"))
                    stats_per_label[label]['error'] += 1
                else:
                    # So sánh với true label (sau khi normalize cả 2)
                    predicted_normalized = normalize_label_for_comparison(predicted_label)
                    is_correct = (predicted_normalized == true_label_normalized)
                    
                    results.append((image_path, label, predicted_label, is_correct, confidence, ""))
                    
                    if is_correct:
                        stats_per_label[label]['correct'] += 1
                    else:
                        stats_per_label[label]['error'] += 1
        
        print(f"   Hoàn thành folder '{label}'")
        print()
    
    print("=" * 60)
    print("KẾT QUẢ THỐNG KÊ")
    
    # Tính tổng số ảnh
    total_correct = sum(stats['correct'] for stats in stats_per_label.values())
    total_no_hand = sum(stats['no_hand'] for stats in stats_per_label.values())
    total_error = sum(stats['error'] for stats in stats_per_label.values())
    
    print(f"\n THỐNG KÊ TỔNG QUAN:")
    print(f"  Tổng số ảnh: {total_images}")
    print(f"  Đã xử lý được: {processed_images}")
    print(f"  Không detect được hand: {total_no_hand}")
    print(f"  Có lỗi khác: {total_error}")
    print(f"  Dự đoán đúng: {total_correct}")
    print(f"  Dự đoán sai: {processed_images - total_correct}")
    
    if processed_images > 0:
        overall_accuracy = (total_correct / processed_images) * 100
        print(f"   Tỷ lệ đúng tổng thể: {overall_accuracy:.2f}%")
    else:
        print(f"    Không có ảnh nào được xử lý thành công")
    
    print(f"\n THỐNG KÊ THEO TỪNG NHÃN:")
    for label in sorted(stats_per_label.keys()):
        stats = stats_per_label[label]
        total = stats['total']
        processed = stats['processed']
        correct = stats['correct']
        no_hand = stats['no_hand']
        error = stats['error']
        
        # Tỷ lệ đúng dựa trên số đã xử lý được (không tính ảnh không detect được hand)
        accuracy_processed = (correct / processed * 100) if processed > 0 else 0.0
        # Tỷ lệ đúng dựa trên tổng số ảnh (tính cả ảnh không detect được)
        accuracy_total = (correct / total * 100) if total > 0 else 0.0
        
        print(f"   {label}:")
        print(f"    Tổng số ảnh: {total}")
        print(f"    Đã xử lý được: {processed} | Không detect: {no_hand} | Lỗi khác: {error}")
        print(f"    Dự đoán đúng: {correct} | Dự đoán sai: {processed - correct}")
        print(f"     Tỷ lệ đúng (dựa trên đã xử lý): {accuracy_processed:.2f}%")
        print(f"     Tỷ lệ đúng (dựa trên tổng số): {accuracy_total:.2f}%")
        print()
    
    # Ghi CSV
    csv_path = os.path.join(script_dir, "thongke.csv")
    print(f"\n Đang ghi kết quả vào CSV: {csv_path}")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Path', 'True Label', 'Predicted Label', 'Is Correct', 'Confidence', 'Error Message'])
        
        for image_path, true_label, predicted_label, is_correct, confidence, error_msg in results:
            writer.writerow([
                image_path,
                true_label,
                predicted_label if predicted_label else "",
                "True" if is_correct else "False",
                f"{confidence:.4f}" if confidence > 0 else "",
                error_msg
            ])
    
    print(f"  Đã ghi {len(results)} dòng vào CSV")
    
    return results, stats_per_label

#  7. Main
if __name__ == "__main__":
    print("THỐNG KÊ GESTURE RECOGNITION")
    print()
    
    # Nhập đường dẫn thư mục labels
    while True:
        labels_dir = input("Nhập đường dẫn thư mục labels: ").strip()
        
        if not labels_dir:
            print("   Vui lòng nhập đường dẫn!")
            continue
        
        # Xử lý đường dẫn (bỏ dấu ngoặc kép nếu có)
        labels_dir = labels_dir.strip('"').strip("'").strip()
        labels_dir = os.path.abspath(labels_dir)
        
        if not os.path.exists(labels_dir):
            print(f"  Đường dẫn không tồn tại: {labels_dir}")
            print("  Vui lòng nhập lại!")
            print()
            continue
        
        if not os.path.isdir(labels_dir):
            print(f"  Đường dẫn không phải là thư mục: {labels_dir}")
            print("  Vui lòng nhập lại!")
            print()
            continue
        
        break
    
    print(f" Đường dẫn hợp lệ: {labels_dir}")
    print()
    
    # Chạy evaluation
    try:
        start_time = time.time()
        evaluate_images(labels_dir)
        end_time = time.time()
        print(f"\n  Thời gian xử lý: {end_time - start_time:.2f} giây")
    except Exception as e:
        print(f"\n LỖI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Đóng landmarker
        try:
            landmarker.close()
        except Exception:
            pass
