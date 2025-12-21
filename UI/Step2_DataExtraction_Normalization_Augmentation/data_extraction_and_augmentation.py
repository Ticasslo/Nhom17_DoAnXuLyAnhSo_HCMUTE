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
6. Lưu vào CSV với NUM_FEATURES features (42 landmarks + 2 Y_hand + 2 X_hand) + label + has_hand + handedness (Left/Right)
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks.python import vision

# 1. ĐỊNH NGHĨA GESTURE TYPES
# Danh sách gesture đối xứng (symmetric)
# Các gesture này có thể flip ảnh để tăng data
# Và các gesture S_* khác trong dataset thực tế
SYMMETRIC_GESTURES = {
    # Gesture điều khiển thiết bị (đối xứng)
    'S_FanOff', 'S_Start', 'S_FanUp', 'S_FanDown',
    'S_FanSpeed1', 'S_FanSpeed2', 'S_FanSpeed3',
    'S_Light1On', 'S_Light1Off', 'S_Light2On', 'S_Light2Off',
}
# Các gesture còn lại là ASYMMETRIC (A_LH_*, A_RH_*)
# Không flip vì tay trái và tay phải có ý nghĩa khác nhau

# CONSTANTS
NUM_LANDMARKS = 21  # Số lượng landmarks cho mỗi hand
# MCP (Metacarpophalangeal) joints: Thumb=2, Index=5, Middle=9, Ring=13, Pinky=17
# MCP đại diện cho hướng của ngón tay, không bị ảnh hưởng bởi việc gập ngón
# Dùng trung bình của 4 ngón (Index, Middle, Ring, Pinky) để đại diện hướng chính của bàn tay
# Trừ thumb vì thumb có hướng khác (vuông góc với các ngón khác)
FINGER_MCP_INDICES = [5, 9, 13, 17]  # Index, Middle, Ring, Pinky MCPs
INDEX_MCP_IDX = 5   # Index finger MCP landmark index
PINKY_MCP_IDX = 17  # Pinky finger MCP landmark index
NUM_FEATURES = 46  # 42 landmarks (21 * 2) + 2 Y_hand + 2 X_hand = 46 features

# 1.5. HÀM TIỀN XỬ LÝ
def tien_xu_ly(img):
    # MỤC TIÊU CHÍNH: CÂN BẰNG DATASET - Giúp đồng nhất dữ liệu Landmarks giữa các ảnh chụp ở điều kiện khác nhau.
    if img is None:
        return None

    # 0. Resize về 640x640: Đồng nhất không gian tọa độ để Landmarks không bị co giãn sai lệch.
    img = cv2.resize(img, (640, 640))

    # 1. Float: Chuyển sang số thực [0, 1] để xử lý chính xác các thay đổi nhỏ về cường độ sáng.
    img = img.astype(np.float32) / 255.0

    # 2. White balance (Cân bằng trắng): 
    # Sửa lỗi ám màu, đưa màu da tay về chuẩn thực tế giúp AI nhận diện đúng đối tượng.
    mean = img.mean(axis=(0, 1), keepdims=True)
    gray = mean.mean()
    scale = np.clip(gray / (mean + 1e-6), 0.9, 1.1)
    img = np.clip(img * scale, 0, 1)
    
    # Bù sáng nhẹ toàn cục (10%) để hỗ trợ MediaPipe thấy rõ các nếp gấp khớp tay.
    img = np.clip(img * 1.1, 0, 1)

    # 3. Gamma thích nghi (Adaptive Gamma):
    # Triệt tiêu sự khác biệt về độ tương phản giữa ảnh sáng và ảnh tối.
    # Đảm bảo hình dạng bộ khung xương (Landmarks) luôn nhất quán bất kể môi trường chụp.
    avg = img.mean()
    gamma = np.clip(1.4 - avg, 0.9, 1.4)
    img = np.power(img, 1/gamma)

    # 4. Denoise (Gaussian Blur 3x3):
    # Loại bỏ các hạt nhiễu kỹ thuật số, giúp các điểm Landmarks đứng yên một chỗ, không bị nhảy lung tung.
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return (img * 255).astype(np.uint8)

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

# 2. HÀM NORMALIZE FEATURES
def normalize_features(landmarks_array):
    """
    Normalize landmarks: relative + scale normalization + orientation features
    PHẢI GIỐNG HỆT với hàm normalize_features() trong classification
    
    Input: landmarks_array shape (21, 2) với (x, y) [đã normalized [0,1] từ MediaPipe]
    Output: NUM_FEATURES features [x0, y0, x1, y1, ..., x20, y20, y_hand_x, y_hand_y, x_hand_x, x_hand_y] đã normalize
    
    Quy trình:
    1. Relative normalization về wrist (landmark 0)
    2. Scale normalization (chia cho max absolute value)
    3. Tính orientation vector (từ wrist đến trung bình của 4 MCP joints: Index, Middle, Ring, Pinky)
    4. Flatten thành NUM_FEATURES features (42 landmarks + 2 Y_hand + 2 X_hand)
    
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
    
    # ORIENTATION FEATURES
    # Tính 2 trục cơ bản của bàn tay để phân biệt hướng và rotation
    # 
    # QUY TRÌNH:
    # 1. Dựng hệ trục bàn tay (hand local coordinate system):
    #    - Y_hand = wrist → mean(MCPs) (trục DỌC của bàn tay)
    #    - X_hand = index_MCP → pinky_MCP (trục NGANG của bàn tay)
    # 2. Normalize cả 2 trục thành unit vectors
    # 3. Lưu vào features [indices 42-45] để model học
    # 
    # TẠI SAO CẦN CẢ 2 TRỤC?
    # - CHỈ Y_hand: Phân biệt Up/Down/Left/Right nhưng KHÔNG biết rotation angle
    # - CẢ Y_hand + X_hand: Phân biệt đầy đủ orientation 2D (hướng + rotation)
    # 
    # VÍ DỤ:
    # - Thumbs Up thẳng: Y=(0,-1), X=(+1,0)
    # - Thumbs Up xoay 45°: Y=(-0.7,-0.7), X=(+0.7,-0.7)
    # → Model học được "Thumbs Up" bất kể rotation angle
    # 
    # LƯU Ý VỀ SYMMETRIC AUGMENTATION:
    # - Khi flip ảnh (horizontal mirror), X_hand sẽ đổi dấu (mirror horizontal)
    # - Model sẽ học CẢ 2 hướng X_hand cho symmetric gestures
    # - Ví dụ: Thumbs Up có thể có X=(+1,0) hoặc X=(-1,0) sau augmentation
    
    # Tính trung bình của 4 MCP joints (Index, Middle, Ring, Pinky)
    mcp_x_avg = np.mean(normalized_x[FINGER_MCP_INDICES])
    mcp_y_avg = np.mean(normalized_y[FINGER_MCP_INDICES])
    
    # Y_hand: vector từ wrist (0,0) đến mean(MCPs) - đây là hướng lên của bàn tay
    y_hand_x = mcp_x_avg  # Wrist = (0,0) sau relative normalization
    y_hand_y = mcp_y_avg
    
    # X_hand: vector từ index_MCP đến pinky_MCP - đây là hướng ngang của bàn tay
    x_hand_x = normalized_x[PINKY_MCP_IDX] - normalized_x[INDEX_MCP_IDX]
    x_hand_y = normalized_y[PINKY_MCP_IDX] - normalized_y[INDEX_MCP_IDX]
    
    EPSILON = 1e-6  # Ngưỡng tối thiểu để tránh chia cho 0
    
    # Normalize Y_hand TRƯỚC (ưu tiên orientation) - đây là trục chính của bàn tay
    y_hand_mag = np.sqrt(y_hand_x**2 + y_hand_y**2)
    if y_hand_mag > EPSILON:
        y_hand_x_normalized = y_hand_x / y_hand_mag
        y_hand_y_normalized = y_hand_y / y_hand_mag
    else:
        # Fallback: nếu không tính được Y_hand (tất cả landmarks trùng nhau), dùng (0, -1) mặc định
        y_hand_x_normalized = 0.0
        y_hand_y_normalized = -1.0
    
    # Normalize X_hand SAU (dựa trên Y_hand đã normalize)
    x_hand_mag = np.sqrt(x_hand_x**2 + x_hand_y**2)
    if x_hand_mag > EPSILON:
        x_hand_x_normalized = x_hand_x / x_hand_mag
        x_hand_y_normalized = x_hand_y / x_hand_mag
    else:
        # Fallback: nếu không tính được X_hand (2 MCPs trùng nhau), dùng vector vuông góc với Y_hand
        # Đảm bảo X_hand vuông góc với Y_hand để tạo hệ trục chuẩn
        x_hand_x_normalized = -y_hand_y_normalized
        x_hand_y_normalized = y_hand_x_normalized
        # Normalize lại vector vuông góc
        x_hand_mag_fallback = np.sqrt(x_hand_x_normalized**2 + x_hand_y_normalized**2)
        if x_hand_mag_fallback > EPSILON:
            x_hand_x_normalized = x_hand_x_normalized / x_hand_mag_fallback
            x_hand_y_normalized = x_hand_y_normalized / x_hand_mag_fallback
        else:
            # Fallback cuối cùng: nếu vẫn không được, dùng (1, 0) mặc định
            x_hand_x_normalized = 1.0
            x_hand_y_normalized = 0.0
    
    # Đảm bảo orientation vectors là unit vectors (kiểm tra lại)
    y_hand_final_mag = np.sqrt(y_hand_x_normalized**2 + y_hand_y_normalized**2)
    if abs(y_hand_final_mag - 1.0) > EPSILON and y_hand_final_mag > EPSILON:
        y_hand_x_normalized = y_hand_x_normalized / y_hand_final_mag
        y_hand_y_normalized = y_hand_y_normalized / y_hand_final_mag
    
    x_hand_final_mag = np.sqrt(x_hand_x_normalized**2 + x_hand_y_normalized**2)
    if abs(x_hand_final_mag - 1.0) > EPSILON and x_hand_final_mag > EPSILON:
        x_hand_x_normalized = x_hand_x_normalized / x_hand_final_mag
        x_hand_y_normalized = x_hand_y_normalized / x_hand_final_mag
    
    # Lưu orientation vectors đã normalize vào features
    # Y_hand: trục dọc của bàn tay (wrist → mean(MCPs)) - hướng lên/xuống
    # X_hand: trục ngang của bàn tay (index_MCP → pinky_MCP) - hướng trái/phải + rotation
    # Cả 2 đã normalize thành unit vectors → dùng trực tiếp
    # Model tự học từ raw orientation values để phân biệt Up/Down/Left/Right + rotation angle
    
    # Flatten thành NUM_FEATURES features (42 landmarks + 2 Y_hand + 2 X_hand)
    feats = np.empty(NUM_FEATURES, dtype=np.float32)
    for i in range(NUM_LANDMARKS):
        feats[2*i] = float(normalized_x[i])
        feats[2*i+1] = float(normalized_y[i])
    
    # Thêm Y_hand features (indices 42, 43) - đã normalize và kiểm tra
    feats[NUM_LANDMARKS * 2] = float(y_hand_x_normalized)  # Index 42: Y_hand X component
    feats[NUM_LANDMARKS * 2 + 1] = float(y_hand_y_normalized)  # Index 43: Y_hand Y component
    
    # Thêm X_hand features (indices 44, 45) - đã normalize và kiểm tra
    feats[NUM_LANDMARKS * 2 + 2] = float(x_hand_x_normalized)  # Index 44: X_hand X component
    feats[NUM_LANDMARKS * 2 + 3] = float(x_hand_y_normalized)  # Index 45: X_hand Y component
    
    return feats

# 3. MEDIAPIPE HAND LANDMARKER SETUP
# Đường dẫn model MediaPipe (.task) dùng chung cho TOÀN project: Nhom17_DoAnXuLyAnhSo_HCMUTE/models/hand_landmarker.task
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # .../Nhom17_DoAnXuLyAnhSo_HCMUTE
HAND_LANDMARKER_MODEL_PATH = os.path.join(project_root, "models", "hand_landmarker.task")

if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
    raise FileNotFoundError(
        f"Khong tim thay model: {HAND_LANDMARKER_MODEL_PATH}"
    )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Tạo HandLandmarker với static image mode
base_options = BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)
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

# 4. DATASET PATH
DATASET_DIR = os.path.join(script_dir, "dataset")

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Khong tim thay thu muc dataset: {DATASET_DIR}")

print(f"[OK] Dataset directory: {DATASET_DIR}")

# 5. DATA EXTRACTION + AUGMENTATION PIPELINE
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
        
        # Áp dụng tiền xử lý
        img = tien_xu_ly(img)
        
        # Convert BGR → RGB (MediaPipe cần RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False  # Tránh copy không cần thiết, tối ưu memory
        
        # Tạo MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # MediaPipe Hand Landmarker detection
        detection_result = landmarker.detect(mp_image)
        del mp_image
        
        #  TRƯỜNG HỢP S_Nothing: KHÔNG có tay 
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
        
        #  CÁC GESTURE KHÁC: cần đúng 1 tay 
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
        
        # VALIDATION: Kiểm tra features có chứa NaN hoặc Inf không
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            msg = f"{gesture_folder}/{img_file}: features chứa NaN hoặc Inf (skip)"
            skip_log.append(msg)
            print(f"    [SKIP] {msg}")
            skipped_images += 1
            continue
        
        data.append(features.tolist())
        labels.append(gesture_folder)
        has_hand_flags.append(1)
        handedness_flags.append(handedness_label)
        gesture_samples += 1
        gesture_original_samples += 1
        processed_images += 1
        
        # DATA AUGMENTATION
        # CHỈ ÁP DỤNG CHO SYMMETRIC GESTURES
        # Asymmetric gestures (A_LH_*, A_RH_*) KHÔNG được augment vì:
        # - Tay trái và tay phải có ý nghĩa khác nhau
        # - FanLeft/FanRight = ngón cái chỉ trái/phải (trục X)
        # - Orientation tính từ 4 MCPs (trục Y) → không phản ánh đúng hướng của thumb
        # - Vậy không cần quan tâm orientation khi augment (vì không augment asymmetric)
        if is_symmetric:
            # Flip ảnh horizontal (mirror theo trục dọc)
            img_flipped = cv2.flip(img, 1)  # 1 = horizontal flip
            
            # Convert BGR → RGB (MediaPipe cần RGB)
            img_rgb_flipped = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB)
            img_rgb_flipped.flags.writeable = False  # Tránh copy không cần thiết, tối ưu memory
            
            # Tạo MediaPipe Image object cho ảnh đã flip
            mp_image_flipped = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb_flipped)
            
            # Detect lại trên ảnh đã flip
            detection_result_flipped = landmarker.detect(mp_image_flipped)
            del mp_image_flipped
            
            # VALIDATION: Kiểm tra kết quả detection trên ảnh flip
            if not detection_result_flipped.hand_landmarks or len(detection_result_flipped.hand_landmarks) == 0:
                # Không detect được tay trên ảnh flip → skip augmentation cho ảnh này
                # (chấp nhận mất một số samples để đảm bảo tính realistic)
                # LƯU Ý: Ảnh gốc đã được xử lý thành công, chỉ augmentation fail
                augmentation_failed_count += 1
                augmentation_failed_log.append(f"{gesture_folder}/{img_file}: flip augmentation failed (0 tay)")
                del img_rgb_flipped  # Cleanup trước khi continue
                continue
            
            if len(detection_result_flipped.hand_landmarks) != 1:
                # Detect được nhiều tay hoặc không đúng 1 tay → skip
                # LƯU Ý: Ảnh gốc đã được xử lý thành công, chỉ augmentation fail
                augmentation_failed_count += 1
                augmentation_failed_log.append(f"{gesture_folder}/{img_file}: flip augmentation failed ({len(detection_result_flipped.hand_landmarks)} tay)")
                del img_rgb_flipped  # Cleanup trước khi continue
                continue
            
            # Extract landmarks từ kết quả detection trên ảnh flip
            hand_landmarks_flipped = detection_result_flipped.hand_landmarks[0]
            landmarks_array_flipped = np.array([[lm.x, lm.y] for lm in hand_landmarks_flipped], dtype=np.float32)
            
            # VALIDATION: Kiểm tra số lượng landmarks
            if len(landmarks_array_flipped) != NUM_LANDMARKS:
                # LƯU Ý: Ảnh gốc đã được xử lý thành công, chỉ augmentation fail
                augmentation_failed_count += 1
                augmentation_failed_log.append(f"{gesture_folder}/{img_file}: flip augmentation failed (landmarks != {NUM_LANDMARKS})")
                del img_rgb_flipped  # Cleanup trước khi continue
                continue
            
            # Normalize features từ landmarks đã detect trên ảnh flip
            features_flipped = normalize_features(landmarks_array_flipped)
            
            # VALIDATION: Kiểm tra features_flipped có chứa NaN hoặc Inf không
            if np.any(np.isnan(features_flipped)) or np.any(np.isinf(features_flipped)):
                # LƯU Ý: Ảnh gốc đã được xử lý thành công, chỉ augmentation fail
                augmentation_failed_count += 1
                augmentation_failed_log.append(f"{gesture_folder}/{img_file}: flip augmentation failed (features chứa NaN/Inf)")
                del img_rgb_flipped  # Cleanup trước khi continue
                continue
            
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
            
            # Cleanup img_rgb_flipped sau khi đã xử lý xong
            del img_rgb_flipped
        
        # Hiển thị progress mỗi 100 ảnh hoặc ảnh cuối cùng
        if img_idx % 100 == 0 or img_idx == len(image_files):
            print(f"    Progress: {img_idx}/{len(image_files)} ảnh đã xử lý...")
    
    # Thông báo kết quả cho gesture này
    if is_symmetric:
        print(f"  [OK] Đã xử lý: {gesture_samples} samples ({gesture_original_samples} gốc + {gesture_augmented_samples} augmented) từ {len(image_files)} ảnh")
    else:
        print(f"  [OK] Đã xử lý: {gesture_samples} samples từ {len(image_files)} ảnh (không augmentation)")

# 6. LƯU DATA VÀO CSV
print(f"\n{'='*60}")
print(f"Dang luu data vao CSV...")

# Tạo DataFrame
feat_cols = [f'feat_{i}' for i in range(NUM_FEATURES)]  # NUM_FEATURES: 42 landmarks + 2 Y_hand + 2 X_hand
df = pd.DataFrame(data, columns=feat_cols)  # NUM_FEATURES cột features (đã normalize)
df['label'] = labels
df['has_hand'] = has_hand_flags
df['handedness'] = handedness_flags

# Lưu vào CSV
output_csv = os.path.join(script_dir, "dataset.csv")
df.to_csv(output_csv, index=False)

# 7. THỐNG KÊ KẾT QUẢ
print(f"{'='*60}")
print(f"[OK] HOAN THANH DATA EXTRACTION + AUGMENTATION")
print(f"[STATS] Thong ke:")
print(f"  - Tong so anh: {total_images}")
print(f"  - Da xu ly: {processed_images} anh")
print(f"  - Da bo qua (skip hoan toan): {skipped_images} anh")
if augmentation_failed_count > 0:
    print(f"  - Augmentation failed: {augmentation_failed_count} lan (anh goc OK nhung flip fail)")
print(f"  - Tong so samples: {len(data)}")
print(f"  - So gesture types: {len(set(labels))}")
print(f"  - Unique labels: {sorted(set(labels))}")
if handedness_mismatch_count > 0:
    print(f"  - Handedness mismatch: {handedness_mismatch_count} samples (đã bỏ qua)")
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
        
# Cleanup
landmarker.close()

