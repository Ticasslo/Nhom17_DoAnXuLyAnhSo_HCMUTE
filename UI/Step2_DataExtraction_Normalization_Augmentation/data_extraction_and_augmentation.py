"""
BƯỚC 2: DATA EXTRACTION + AUGMENTATION

Mục đích:
- Trích xuất hand landmarks từ ảnh đã chụp
- Áp dụng augmentation cho symmetric gestures (flip ảnh)
- Normalize features (relative + scale normalization)
- Lưu vào CSV file

Quy trình:
1. Định nghĩa gesture types (symmetric vs asymmetric)
2. Load MediaPipe Hand Landmarker
3. Duyệt qua từng ảnh trong dataset
4. Extract landmarks và normalize
5. Augmentation cho symmetric gestures (flip ảnh)
6. Lưu vào CSV với NUM_FEATURES features (42 landmarks + 2 orientation) + label + has_hand + handedness
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks.python import vision

# ===============================================================
# 1. ĐỊNH NGHĨA GESTURE TYPES
# ===============================================================

# Danh sách gesture đối xứng (symmetric)
# Các gesture này có thể flip ảnh để tăng data
# Theo yêu cầu: S_A đến S_Z, S_Ok, S_ThumbsUp, S_Space, S_Del, S_Nothing
# Và các gesture S_* khác trong dataset thực tế
SYMMETRIC_GESTURES = {
    # Chữ cái đối xứng (theo yêu cầu)
    'S_A', 'S_B', 'S_C', 'S_D', 'S_E', 'S_F', 'S_G', 'S_H',
    'S_I', 'S_J', 'S_K', 'S_L', 'S_M', 'S_N', 'S_O', 'S_P',
    'S_Q', 'S_R', 'S_S', 'S_T', 'S_U', 'S_V', 'S_W', 'S_X',
    'S_Y', 'S_Z',
    # Gesture đối xứng khác
    'S_Ok', 'S_ThumbsUp',
    'S_Space', 'S_Del', 'S_Nothing',
    # Gesture điều khiển thiết bị (đối xứng)
    'S_FanOff', 'S_Start', 'S_FanUp', 'S_FanDown',
    'S_FanSpeed1', 'S_FanSpeed2', 'S_FanSpeed3',
    'S_Light1On', 'S_Light1Off', 'S_Light2On', 'S_Light2Off',
}

# Các gesture còn lại là ASYMMETRIC (A_LH_*, A_RH_*)
# Không flip vì tay trái và tay phải có ý nghĩa khác nhau

# ===============================================================
# CONSTANTS
# ===============================================================
NUM_LANDMARKS = 21  # Số lượng landmarks cho mỗi hand
# MCP (Metacarpophalangeal) joints: Thumb=2, Index=5, Middle=9, Ring=13, Pinky=17
# MCP đại diện cho hướng của ngón tay, không bị ảnh hưởng bởi việc gập ngón
# Dùng trung bình của 4 ngón (Index, Middle, Ring, Pinky) để đại diện hướng chính của bàn tay
# Trừ thumb vì thumb có hướng khác (vuông góc với các ngón khác)
FINGER_MCP_INDICES = [5, 9, 13, 17]  # Index, Middle, Ring, Pinky MCPs
NUM_FEATURES = 44  # 42 landmarks + 2 orientation

# ===============================================================
# VALIDATION: Kiểm tra handedness có khớp với label không
# ===============================================================

def validate_handedness(gesture_folder, detected_handedness):
    # Chỉ validate cho asymmetric gestures
    if not (gesture_folder.startswith('A_LH_') or gesture_folder.startswith('A_RH_')):
        return True, None, None  # Symmetric gestures không cần validate
    
    # Xác định expected hand từ folder name
    if gesture_folder.startswith('A_LH_'):
        expected_hand = 'Left'
    elif gesture_folder.startswith('A_RH_'):
        expected_hand = 'Right'
    else:
        return True, None, None  # Không phải asymmetric
    
    # Nếu không detect được handedness, không thể validate
    if detected_handedness is None:
        return False, expected_hand, "Không detect được handedness"
    
    # Normalize detected handedness (có thể là "Left", "Right", hoặc "1 Left", "1 Right")
    detected_hand = str(detected_handedness).strip()
    if 'Left' in detected_hand:
        detected_hand = 'Left'
    elif 'Right' in detected_hand:
        detected_hand = 'Right'
    else:
        return False, expected_hand, f"Handedness không hợp lệ: {detected_handedness}"
    
    # Kiểm tra khớp
    if detected_hand == expected_hand:
        return True, expected_hand, None
    else:
        return False, expected_hand, f"Không khớp: expected {expected_hand}, detected {detected_hand}"

# ===============================================================
# 2. HÀM NORMALIZE FEATURES
# ===============================================================

def normalize_features(landmarks_array):
    """
    Normalize landmarks: relative + scale normalization + orientation features
    PHẢI GIỐNG HỆT với hàm normalize_features() trong tkinter_template_detection_classification.py!
    
    Input: landmarks_array shape (21, 2) với (x, y) [đã normalized [0,1] từ MediaPipe]
    Output: NUM_FEATURES features [x0, y0, x1, y1, ..., x20, y20, orientation_x, orientation_y] đã normalize
    
    Quy trình:
    1. Relative normalization về wrist (landmark 0)
    2. Scale normalization (chia cho max absolute value)
    3. Tính orientation vector (từ wrist đến trung bình của 4 MCP joints: Index, Middle, Ring, Pinky)
    4. Flatten thành NUM_FEATURES features (42 landmarks + 2 orientation)
    
    LƯU Ý: Thêm orientation features để phân biệt hướng (lên/xuống/trái/phải)
    """
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
    
    # ========== ORIENTATION FEATURES ==========
    # Tính vector hướng từ wrist (0) đến TRUNG BÌNH của các MCP joints
    # MCP (Metacarpophalangeal joint) = điểm gốc của ngón tay
    # 
    # TẠI SAO DÙNG TRUNG BÌNH CỦA 4 MCPs (Index, Middle, Ring, Pinky)?
    # - Đại diện cho HƯỚNG CHÍNH của bàn tay, không phụ thuộc vào 1 ngón
    # - Giải quyết mâu thuẫn: 4 ngón hướng lên nhưng ngón trỏ chỉ phải
    #   → Trung bình sẽ là "lên trên" (đúng với hướng chính của bàn tay)
    # - Ổn định hơn dùng 1 ngón, chính xác hơn tip
    # - Trừ thumb vì thumb có hướng khác (vuông góc với các ngón khác)
    # 
    # Ví dụ 1: Tay chỉ sang phải, ngón trỏ gập vuông góc lên trên
    # - Index MCP: orientation_x > 0 (phải)
    # - Trung bình 4 MCPs: orientation_x > 0 (phải) ✅ ĐÚNG
    # 
    # Ví dụ 2: 4 ngón hướng lên, ngón trỏ chỉ phải
    # - Index MCP: orientation_x > 0 (phải)
    # - Middle/Ring/Pinky MCPs: orientation_y < 0 (lên)
    # - Trung bình: orientation_y < 0 (lên) ✅ ĐÚNG (hướng chính của bàn tay)
    
    # Tính trung bình của 4 MCP joints (Index, Middle, Ring, Pinky)
    mcp_x_avg = np.mean([normalized_x[i] for i in FINGER_MCP_INDICES])
    mcp_y_avg = np.mean([normalized_y[i] for i in FINGER_MCP_INDICES])
    
    orientation_x = mcp_x_avg
    orientation_y = mcp_y_avg
    
    # Normalize orientation vector (đảm bảo trong khoảng [-1, 1])
    orientation_magnitude = np.sqrt(orientation_x**2 + orientation_y**2)
    if orientation_magnitude > 0:
        orientation_x = orientation_x / orientation_magnitude
        orientation_y = orientation_y / orientation_magnitude
    # Nếu magnitude = 0 (không có hướng), giữ nguyên (0, 0)
    # ==============================================
    
    # Flatten thành NUM_FEATURES features (42 landmarks + 2 orientation)
    feats = np.empty(NUM_FEATURES, dtype=np.float32)
    for i in range(NUM_LANDMARKS):
        feats[2*i] = float(normalized_x[i])
        feats[2*i+1] = float(normalized_y[i])
    
    # Thêm orientation features ở cuối (indices 42, 43)
    feats[NUM_LANDMARKS * 2] = float(orientation_x)  # Index 42
    feats[NUM_LANDMARKS * 2 + 1] = float(orientation_y)  # Index 43
    
    return feats

# ===============================================================
# 3. MEDIAPIPE HAND LANDMARKER SETUP
# ===============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
HAND_LANDMARKER_MODEL_PATH = os.path.join(script_dir, "hand_landmarker.task")

if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
    raise FileNotFoundError(
        f"Khong tim thay model MediaPipe: {HAND_LANDMARKER_MODEL_PATH}\n"
        f"Vui long dam bao file hand_landmarker.task co trong thu muc nay."
    )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Tạo HandLandmarker với static image mode
base_options = BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)
# Nới lỏng ngưỡng để giảm bỏ sót tay
# Ảnh đầu vào đã 640x640 nên không cần resize thêm
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,  # IMAGE mode cho static images
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

print("[OK] MediaPipe Hand Landmarker da duoc khoi tao")

# ===============================================================
# 4. DATASET PATH
# ===============================================================

DATASET_DIR = os.path.join(script_dir, "dataset")

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Khong tim thay thu muc dataset: {DATASET_DIR}")

print(f"[OK] Dataset directory: {DATASET_DIR}")

# ===============================================================
# 4.5. VALIDATION HANDEDNESS
# ===============================================================
# LUÔN LUÔN bỏ qua các sample có handedness không khớp với label folder
# (A_LH_* phải có handedness = Left, A_RH_* phải có handedness = Right)

# ===============================================================
# 5. DATA EXTRACTION + AUGMENTATION PIPELINE
# ===============================================================

data = []
labels = []
has_hand_flags = []
handedness_flags = []

# Đếm số lượng ảnh đã xử lý
total_images = 0
processed_images = 0
skipped_images = 0  # Ảnh bị skip hoàn toàn (không xử lý được)
augmentation_failed_count = 0  # Augmentation fail (ảnh gốc OK nhưng flip fail)
handedness_mismatch_count = 0  # Đếm số lượng mismatch
handedness_mismatch_details = []  # Chi tiết các mismatch
# Log các ảnh bị skip để debug (in hết ra console, không ghi file)
skip_log = []
augmentation_failed_log = []  # Log các augmentation failures

# Duyệt qua từng gesture folder
gesture_folders = sorted([f for f in os.listdir(DATASET_DIR) 
                         if os.path.isdir(os.path.join(DATASET_DIR, f))])

print(f"\n{'='*60}")
print(f"Bat dau xu ly {len(gesture_folders)} gesture folders...")
print(f"{'='*60}\n")

for gesture_folder in gesture_folders:
    gesture_path = os.path.join(DATASET_DIR, gesture_folder)
    
    if not os.path.isdir(gesture_path):
        continue
    
    # Kiểm tra gesture có đối xứng không
    is_symmetric = gesture_folder in SYMMETRIC_GESTURES
    
    print(f"[PROCESSING] {gesture_folder} | Symmetric: {is_symmetric}")
    
    # Đếm số ảnh trong folder
    image_files = [f for f in os.listdir(gesture_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images += len(image_files)
    
    gesture_samples = 0
    gesture_original_samples = 0
    gesture_augmented_samples = 0
    
    # Duyệt qua từng ảnh trong folder
    for img_idx, img_file in enumerate(sorted(image_files), 1):
        img_path = os.path.join(gesture_path, img_file)
        
        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            msg = f"{gesture_folder}/{img_file}: lỗi đọc ảnh"
            skip_log.append(msg)
            print(f"  [WARNING] {msg}")
            skipped_images += 1
            continue
        
        # Convert BGR → RGB (MediaPipe cần RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Tạo MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # MediaPipe Hand Landmarker detection
        detection_result = landmarker.detect(mp_image)
        
        # ========== TRƯỜNG HỢP S_Nothing: KHÔNG có tay ==========
        if gesture_folder == 'S_Nothing':
            if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
                # Có tay thì bỏ (vì nhãn này là không tay)
                skipped_images += 1
                continue
            
            # Không tay → gán vector zero NUM_FEATURES chiều + has_hand=0
            features = [0.0] * NUM_FEATURES
            data.append(features)
            labels.append(gesture_folder)
            has_hand_flags.append(0)
            handedness_flags.append(None)
            gesture_samples += 1
            processed_images += 1
            continue  # sang ảnh kế tiếp
        
        # ========== CÁC GESTURE KHÁC: cần đúng 1 tay ==========
        if not detection_result.hand_landmarks or len(detection_result.hand_landmarks) == 0:
            msg = f"{gesture_folder}/{img_file}: 0 tay (skip)"
            skip_log.append(msg)
            print(f"    [SKIP] {msg}")
            skipped_images += 1
            continue
        
        if len(detection_result.hand_landmarks) != 1:
            msg = f"{gesture_folder}/{img_file}: {len(detection_result.hand_landmarks)} tay (skip, cần đúng 1)"
            skip_log.append(msg)
            print(f"    [SKIP] {msg}")
            skipped_images += 1
            continue  # bỏ ảnh có 0 tay hoặc >1 tay
        
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Lấy handedness từ MediaPipe (Left/Right nếu có)
        handedness_label = None
        if detection_result.handedness and len(detection_result.handedness) > 0:
            entry = detection_result.handedness[0]
            # Xử lý an toàn: entry có thể là list/tuple hoặc object trực tiếp
            if isinstance(entry, (list, tuple)) and len(entry) > 0:
                cat = entry[0]
            else:
                cat = entry
            
            # Đọc category_name và score (hỗ trợ nhiều version MediaPipe)
            name = getattr(cat, "category_name", None) or getattr(cat, "label", None) or "Hand"
            handedness_label = name
        
        # VALIDATION: Kiểm tra handedness có khớp với label không
        # LUÔN LUÔN BỎ QUA các sample không khớp
        is_valid, expected_hand, mismatch_reason = validate_handedness(gesture_folder, handedness_label)
        
        if not is_valid:
            # LUÔN LUÔN bỏ qua sample không khớp
            mismatch_info = f"{gesture_folder}: {mismatch_reason} (file: {img_file})"
            handedness_mismatch_details.append(mismatch_info)
            handedness_mismatch_count += 1
            skipped_images += 1
            skip_log.append(f"{gesture_folder}/{img_file}: handedness mismatch ({mismatch_reason})")
            print(f"    [SKIP] {mismatch_info}")
            continue
        
        # Extract + normalize từ ảnh gốc
        # Convert landmarks -> ndarray (21,2) rồi normalize
        landmarks_array = np.array([[lm.x, lm.y] for lm in hand_landmarks], dtype=np.float32)
        
        # VALIDATION: Kiểm tra số lượng landmarks
        if len(landmarks_array) != NUM_LANDMARKS:
            msg = f"{gesture_folder}/{img_file}: landmarks != {NUM_LANDMARKS} (got {len(landmarks_array)})"
            skip_log.append(msg)
            print(f"    [SKIP] {msg}")
            skipped_images += 1
            continue
        
        features = normalize_features(landmarks_array)
        
        data.append(features.tolist())
        labels.append(gesture_folder)
        has_hand_flags.append(1)
        handedness_flags.append(handedness_label)
        gesture_samples += 1
        gesture_original_samples += 1
        processed_images += 1
        
        # ========== DATA AUGMENTATION ==========
        # CHỈ ÁP DỤNG CHO SYMMETRIC GESTURES
        if is_symmetric:
            # Flip ảnh horizontal (mirror theo trục dọc)
            img_flipped = cv2.flip(img, 1)  # 1 = horizontal flip
            
            # Convert BGR → RGB (MediaPipe cần RGB)
            img_rgb_flipped = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB)
            
            # Tạo MediaPipe Image object cho ảnh đã flip
            mp_image_flipped = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb_flipped)
            
            # Detect lại trên ảnh đã flip
            detection_result_flipped = landmarker.detect(mp_image_flipped)
            
            # VALIDATION: Kiểm tra kết quả detection trên ảnh flip
            if not detection_result_flipped.hand_landmarks or len(detection_result_flipped.hand_landmarks) == 0:
                # Không detect được tay trên ảnh flip → skip augmentation cho ảnh này
                # (chấp nhận mất một số samples để đảm bảo tính realistic)
                # LƯU Ý: Ảnh gốc đã được xử lý thành công, chỉ augmentation fail
                augmentation_failed_count += 1
                augmentation_failed_log.append(f"{gesture_folder}/{img_file}: flip augmentation failed (0 tay)")
                continue
            
            if len(detection_result_flipped.hand_landmarks) != 1:
                # Detect được nhiều tay hoặc không đúng 1 tay → skip
                # LƯU Ý: Ảnh gốc đã được xử lý thành công, chỉ augmentation fail
                augmentation_failed_count += 1
                augmentation_failed_log.append(f"{gesture_folder}/{img_file}: flip augmentation failed ({len(detection_result_flipped.hand_landmarks)} tay)")
                continue
            
            # Extract landmarks từ kết quả detection trên ảnh flip
            hand_landmarks_flipped = detection_result_flipped.hand_landmarks[0]
            landmarks_array_flipped = np.array([[lm.x, lm.y] for lm in hand_landmarks_flipped], dtype=np.float32)
            
            # VALIDATION: Kiểm tra số lượng landmarks
            if len(landmarks_array_flipped) != NUM_LANDMARKS:
                # LƯU Ý: Ảnh gốc đã được xử lý thành công, chỉ augmentation fail
                augmentation_failed_count += 1
                augmentation_failed_log.append(f"{gesture_folder}/{img_file}: flip augmentation failed (landmarks != {NUM_LANDMARKS})")
                continue
            
            # Normalize features từ landmarks đã detect trên ảnh flip
            features_flipped = normalize_features(landmarks_array_flipped)
            
            # Lấy handedness từ kết quả detection trên ảnh flip
            handedness_label_flip = None
            if detection_result_flipped.handedness and len(detection_result_flipped.handedness) > 0:
                entry = detection_result_flipped.handedness[0]
                if isinstance(entry, (list, tuple)) and len(entry) > 0:
                    cat = entry[0]
                else:
                    cat = entry
                name = getattr(cat, "category_name", None) or getattr(cat, "label", None) or "Hand"
                handedness_label_flip = name
            
            # Lưu augmented sample với CÙNG label
            data.append(features_flipped.tolist())
            labels.append(gesture_folder)
            has_hand_flags.append(1)
            handedness_flags.append(handedness_label_flip)
            gesture_samples += 1  # Đếm thêm 1 sample từ augmentation
            gesture_augmented_samples += 1
        
        # Hiển thị progress mỗi 100 ảnh hoặc ảnh cuối cùng
        if img_idx % 100 == 0 or img_idx == len(image_files):
            print(f"    Progress: {img_idx}/{len(image_files)} ảnh đã xử lý...")
    
    # Thông báo kết quả cho gesture này
    if is_symmetric:
        print(f"  [OK] Đã xử lý: {gesture_samples} samples ({gesture_original_samples} gốc + {gesture_augmented_samples} augmented) từ {len(image_files)} ảnh")
    else:
        print(f"  [OK] Đã xử lý: {gesture_samples} samples từ {len(image_files)} ảnh (không augmentation)")

# ===============================================================
# 6. LƯU DATA VÀO CSV
# ===============================================================

print(f"\n{'='*60}")
print(f"Dang luu data vao CSV...")
print(f"{'='*60}\n")

# Tạo DataFrame
feat_cols = [f'feat_{i}' for i in range(NUM_FEATURES)]  # NUM_FEATURES: 42 landmarks + 2 orientation
df = pd.DataFrame(data, columns=feat_cols)  # NUM_FEATURES cột features (đã normalize)
df['label'] = labels
df['has_hand'] = has_hand_flags
df['handedness'] = handedness_flags

# Lưu vào CSV
output_csv = os.path.join(script_dir, "dataset.csv")
df.to_csv(output_csv, index=False)

# ===============================================================
# 7. THỐNG KÊ KẾT QUẢ
# ===============================================================

print(f"{'='*60}")
print(f"[OK] HOAN THANH DATA EXTRACTION + AUGMENTATION")
print(f"{'='*60}")
print(f"[STATS] Thong ke:")
print(f"  - Tong so anh: {total_images}")
print(f"  - Da xu ly: {processed_images} anh")
print(f"  - Da bo qua (skip hoan toan): {skipped_images} anh")
if augmentation_failed_count > 0:
    print(f"  - ⚠️  Augmentation failed: {augmentation_failed_count} lan (anh goc OK nhung flip fail)")
print(f"  - Tong so samples: {len(data)}")
print(f"  - So gesture types: {len(set(labels))}")
print(f"  - Unique labels: {sorted(set(labels))}")
if handedness_mismatch_count > 0:
    print(f"  - ⚠️  Handedness mismatch: {handedness_mismatch_count} samples (đã bỏ qua)")
print(f"\n[FILE] File CSV da luu: {output_csv}")
print(f"{'='*60}\n")

# Thống kê chi tiết theo gesture
print("[DETAIL] Chi tiet theo gesture:")
for label in sorted(set(labels)):
    count = labels.count(label)
    is_sym = label in SYMMETRIC_GESTURES
    print(f"  - {label}: {count} samples {'(symmetric)' if is_sym else '(asymmetric)'}")

# Hiển thị chi tiết các mismatch nếu có
if handedness_mismatch_count > 0:
    print(f"\n[WARNING] Handedness mismatch: {handedness_mismatch_count} samples:")
    for detail in handedness_mismatch_details:
        print(f"  - {detail}")

# In toàn bộ danh sách ảnh bị skip
if skip_log:
    print(f"\n[WARNING] Tổng số ảnh bị skip (skip hoàn toàn): {len(skip_log)}")
    for line in skip_log:
        print(f"  - {line}")

# In danh sách augmentation failures
if augmentation_failed_log:
    print(f"\n[INFO] Tổng số augmentation failures: {len(augmentation_failed_log)}")
    print(f"  (Ảnh gốc đã được xử lý thành công, chỉ augmentation fail)")
    for line in augmentation_failed_log:
        print(f"  - {line}")

print(f"\n{'='*60}\n")

# Cleanup
landmarker.close()

