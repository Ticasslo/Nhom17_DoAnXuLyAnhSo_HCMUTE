import os
import time
import cv2
import warnings
import threading
from queue import Queue, Empty, Full
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import scrolledtext
import pickle
from collections import deque, Counter
import sys

import mediapipe as mp
from mediapipe.tasks.python import vision
import tensorflow as tf

# MQTT Publisher (dùng để publish lệnh qua MQTT)
from mqtt_publisher import init_mqtt_publisher, publish_gesture_command, get_mqtt_publisher

# Giảm warning log
warnings.filterwarnings("ignore", category=UserWarning)

# 0. CONSTANTS
# Label prefixes
ASYMMETRIC_PREFIX_RH = "A_RH_"
ASYMMETRIC_PREFIX_LH = "A_LH_"
SYMMETRIC_PREFIX = "S_"

# MediaPipe landmarks
NUM_LANDMARKS = 21  # Số lượng landmarks cho mỗi hand
# MCP (Metacarpophalangeal) joints: Thumb=2, Index=5, Middle=9, Ring=13, Pinky=17
# MCP đại diện cho hướng của ngón tay, không bị ảnh hưởng bởi việc gập ngón
# Dùng trung bình của 4 ngón (Index, Middle, Ring, Pinky) để đại diện hướng chính của bàn tay
# Trừ thumb vì thumb có hướng khác (vuông góc với các ngón khác)
FINGER_MCP_INDICES = [5, 9, 13, 17]  # Index, Middle, Ring, Pinky MCPs
INDEX_MCP_IDX = 5   # Index finger MCP landmark index
PINKY_MCP_IDX = 17  # Pinky finger MCP landmark index
NUM_FEATURES = 46  # 42 landmarks (21 * 2) + 2 Y_hand + 2 X_hand = 46 features

# 1. CONFIGURATION
# Camera settings - ESP32 Camera Stream
# Thay đổi IP và port theo ESP32 của bạn
# Ví dụ: "http://192.168.1.100:80/stream" hoặc "http://esp32-cam.local:80/stream"
# LƯU Ý: Test URL bằng browser trước: mở http://esp32-cam.local/stream
# Nếu thấy video → URL đúng. Nếu không thấy → ESP32 chưa sẵn sàng hoặc URL sai
SOURCE = "http://esp32-cam.local/stream"  # ESP32 camera stream URL (MJPEG multipart stream)

# Performance settings
NUM_HANDS = 1  # Chỉ 1 tay (ESP32 thường chỉ detect 1 tay)
MIN_DETECTION_CONFIDENCE = 0.7  # Ngưỡng cho palm detector (BlazePalm)
MIN_PRESENCE_CONFIDENCE = 0.6   # Ngưỡng để trigger re-detection (thấp hơn = re-detect thường xuyên hơn)
MIN_TRACKING_CONFIDENCE = 0.6   # Ngưỡng cho hand tracking (landmark model)

# Filtering thresholds
HAND_MIN_AREA_RATIO = 0.0025   # ~% diện tích frame (bỏ box quá nhỏ)
HAND_MAX_AREA_RATIO = 0.45     # ~% diện tích frame (bỏ box quá lớn)
HANDEDNESS_SCORE_THRESHOLD = 0.75  # Ngưỡng confidence tối thiểu cho handedness
GESTURE_REJECT_THRESHOLD = 0.85  # Ngưỡng reject cứng: dưới ngưỡng này coi như Unknown
# Filter TRƯỚC khi vào GestureVoter - reject sớm để hiệu quả hơn
# - Nếu confidence < 0.85 → reject cứng, KHÔNG vào GestureVoter (hiển thị "Unknown")
# - Nếu confidence >= 0.85 → vào GestureVoter để xử lý với voting
# - Tăng lên (0.88-0.92) → khắt khe hơn, ổn định hơn
# - Giảm xuống (0.80-0.83) → dễ chấp nhận hơn nhưng có thể nhận cả predictions không chắc

# Entropy threshold (kiểm tra độ "phẳng" của probability distribution)
# Entropy cao = phân phối phẳng = nhiều class có xác suất gần nhau = không chắc → reject
# Entropy thấp = phân phối tập trung = một class chiếm ưu thế = chắc → accept
ENTROPY_THRESHOLD = 2.0  # Ngưỡng entropy tối đa (entropy > threshold → reject vì không chắc)

# Orientation filtering thresholds
# Orientation (hướng bàn tay) được tính từ 2 trục:
# - Y_hand: vector từ wrist đến trung bình của 4 MCP joints (Index, Middle, Ring, Pinky) - trục DỌC
# - X_hand: vector từ index_MCP đến pinky_MCP - trục NGANG
# Magnitude = độ dài vector (sau khi normalize về [0,1] thì magnitude = sqrt(x^2 + y^2))
# Magnitude nhỏ → landmarks quá gần nhau → orientation không ổn định → prediction không đáng tin

ORIENTATION_MIN_MAGNITUDE = 0.15  # Ngưỡng magnitude tối thiểu cho Y_hand và X_hand (reject khi orientation không ổn định)
# Cả Y_hand và X_hand phải có magnitude ≥ 0.15 mới được chấp nhận
# - Tăng lên (0.20-0.25) → khắt khe hơn, reject nhiều hơn, ổn định hơn nhưng có thể bỏ sót gesture hợp lệ
# - Giảm xuống (0.10-0.12) → dễ chấp nhận hơn, ít reject hơn nhưng có thể nhận cả orientation không ổn định
# Giá trị 0.15 là cân bằng tốt: đủ để filter noise nhưng không quá khắt khe

# GestureVoter settings
# GestureVoter là hệ thống voting theo thời gian để ổn định gesture prediction
# Mỗi frame, nếu prediction hợp lệ → thêm vào votes, sau đó kiểm tra 3 điều kiện:
# 1. Đủ số votes đầu vào (min_votes) - ĐIỀU KIỆN ĐẦU VÀO
# 2. Đủ thời gian giữ tay (acceptance_time) - ĐIỀU KIỆN THỜI GIAN
# 3. Đủ tỷ lệ đồng ý (vote_threshold) - ĐIỀU KIỆN TỶ LỆ

GESTURE_VOTER_MIN_VOTES = 7  # Số votes tối thiểu cần có (ĐIỀU KIỆN ĐẦU VÀO)
# Phải có ít nhất 7 votes mới BẮT ĐẦU kiểm tra các điều kiện khác
# - Đây là điều kiện đầu vào: chưa đủ 7 votes → không kiểm tra gì cả
# - Tăng lên (10-15) → cần nhiều frames hơn, chậm hơn nhưng ổn định hơn
# - Giảm xuống (4-6) → phản ứng nhanh hơn nhưng dễ nhạy cảm với noise
# Ví dụ: FPS 30 → 7 votes = 0.23s, FPS 10 → 7 votes = 0.7s

GESTURE_VOTER_ACCEPTANCE_TIME = 1.2  # Thời gian PHẢI GIỮ TAY (giây) - ĐIỀU KIỆN THỜI GIAN
# Phải giữ tay trong ít nhất 1.2s từ vote đầu tiên mới được chấp nhận gesture
# - Đây là điều kiện thời gian: người dùng phải giữ tay đủ lâu (tránh false positive ngắn)
# - Tăng lên (1.5-2.0) → phải giữ tay lâu hơn, ổn định hơn, ít false positive
# - Giảm xuống (0.5-0.8) → giữ tay ngắn hơn, phản ứng nhanh nhưng dễ nhạy cảm với noise
# LƯU Ý: Phải ≤ GESTURE_VOTER_VOTE_LIFETIME để đảm bảo vote đầu tiên không bị xóa trước khi đạt acceptance_time

GESTURE_VOTER_VOTE_LIFETIME = 2.0  # Thời gian XÉT HỢP LỆ votes (giây) - ĐIỀU KIỆN XÉT
# Votes cũ hơn 2.0s sẽ bị xóa tự động (chỉ xét votes trong 2.0s gần nhất)
# - Đây là điều kiện xét hợp lệ: chỉ tính votes trong khoảng thời gian này
# - 80% (vote_threshold) được tính dựa vào TẤT CẢ votes trong 2.0s này
# - Cao hơn (2.5-3.0) → xét votes lâu hơn, ổn định hơn nhưng phản ứng chậm khi đổi gesture
# - Thấp hơn (1.5-1.8) → xét votes ngắn hơn, phản ứng nhanh nhưng dễ nhạy cảm với noise
# LƯU Ý: Phải ≥ GESTURE_VOTER_ACCEPTANCE_TIME để đảm bảo vote đầu tiên không bị xóa trước khi đạt acceptance_time

GESTURE_VOTER_VOTE_THRESHOLD = 0.80  # Tỷ lệ votes đồng ý tối thiểu (80%) - ĐIỀU KIỆN TỶ LỆ
# Gesture phải chiếm ≥ 80% tổng số votes (trong vote_lifetime) mới được chấp nhận
# - Đây là điều kiện tỷ lệ: tính dựa vào TẤT CẢ votes trong vote_lifetime (2.0s), KHÔNG phải chỉ trong acceptance_time
# - Ví dụ: có 10 votes trong 2.0s, 8 votes là "FanSpeed1" → 80% → chấp nhận
# - Tăng lên (0.85-0.90) → khó chấp nhận hơn, ổn định hơn, ít false positive
# - Giảm xuống (0.70-0.75) → dễ chấp nhận hơn, phản ứng nhanh nhưng dễ nhạy cảm

# Display settings
PRINT_EVERY_N_FRAMES = 200
WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 660

# EMA Smoothing settings
ENABLE_EMA_SMOOTHING = True  # Enable Exponential Moving Average smoothing
EMA_ALPHA = 0.5  # Smoothing factor (0.1=max smooth, 1.0=no smooth).

# Queue settings
FRAME_BUFFER_SIZE = 1
DETECTION_BUFFER_SIZE = 1

DETECTION_SKIP_FRAMES = 1  # Số frame bỏ qua giữa các lần detection (0 = detect mọi frame)


# 2. Load Gesture Model & Metadata
script_dir = os.path.dirname(os.path.abspath(__file__))

# project_root = .../Nhom17_DoAnXuLyAnhSo_HCMUTE
project_root = os.path.dirname(os.path.dirname(script_dir))

# Model TF SavedModel + metadata luôn đặt trong: Nhom17_DoAnXuLyAnhSo_HCMUTE/models/SavedModel/
GESTURE_MODEL_PATH = os.path.join(project_root, "models", "SavedModel", "saved_model_best")
METADATA_PATH = os.path.join(project_root, "models", "SavedModel", "metadata.pkl")

# Load model
if not os.path.exists(GESTURE_MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model TF: {GESTURE_MODEL_PATH}")
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Không tìm thấy metadata: {METADATA_PATH}")

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
    
    # FORMAT:
    # {
    #   'labels': ['A_LH_FanLeft', 'A_LH_FanRight', ..., 'S_Start'],
    #   'num_features': 46,
    #   'feature_columns': ['feat_0', 'feat_1', ...],
    #   'tf_version': '2.16.1',
    #   'format': 'SavedModel',
    # }
    
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata.pkl không phải dict! Type: {type(metadata)}")
    
    # Kiểm tra 'labels'
    if 'labels' not in metadata:
        raise ValueError(f"metadata.pkl không có 'labels'!")
    
    labels_list = metadata['labels']
    if not isinstance(labels_list, list):
        raise ValueError(f"metadata['labels'] phải là list, nhận được: {type(labels_list)}")
    
    if len(labels_list) == 0:
        raise ValueError("metadata['labels'] không được rỗng!")
    
    # Tạo label_mapping: {0: label0, 1: label1, ...}
    label_mapping = {i: label for i, label in enumerate(labels_list)}
    
    # Extract symmetric gestures từ labels (các label bắt đầu bằng 'S_')
    SYMMETRIC_GESTURES = set([label for label in labels_list if label.startswith(SYMMETRIC_PREFIX)])
    
    num_classes = len(labels_list)
    
    # Hiển thị thông tin metadata
    if 'num_features' in metadata:
        print(f"  → Num features: {metadata['num_features']}")
    if 'tf_version' in metadata:
        print(f"  → TF version: {metadata['tf_version']}")
    if 'format' in metadata:
        print(f"  → Format: {metadata['format']}")
    
    print(f"  → Loaded {num_classes} labels")
    print(f"  → Symmetric gestures: {len(SYMMETRIC_GESTURES)}")
print(f"  → Model ready: {num_classes} classes")


# 2.1. GestureVoter Class
class GestureVoter:
    def __init__(self,
                 acceptance_time=GESTURE_VOTER_ACCEPTANCE_TIME,
                 vote_lifetime=GESTURE_VOTER_VOTE_LIFETIME, 
                 vote_threshold=GESTURE_VOTER_VOTE_THRESHOLD,
                 min_votes=GESTURE_VOTER_MIN_VOTES):
        
        # Validation: acceptance_time phải ≤ vote_lifetime
        # Nếu không, vote đầu tiên sẽ bị xóa trước khi đạt acceptance_time
        if acceptance_time > vote_lifetime:
            raise ValueError(
                f"acceptance_time ({acceptance_time}s) phải ≤ vote_lifetime ({vote_lifetime}s). "
            )
        
        self.acceptance_time = acceptance_time
        self.vote_lifetime = vote_lifetime
        self.vote_threshold = vote_threshold
        self.min_votes = min_votes
        self.votes = deque()
        self.current_gesture = None
        self.current_confidence = 0.0
    
    def vote(self, prediction, confidence):     
        current_time = time.time()
        
        # Kiểm tra nếu gesture thay đổi: reset votes và current_gesture
        # Điều này đảm bảo gesture mới bắt đầu từ đầu, không bị ảnh hưởng bởi votes cũ
        if self.current_gesture is not None and self.current_gesture != prediction:
            # Gesture thay đổi → reset hoàn toàn để bắt đầu gesture mới
            self.votes.clear()
            self.current_gesture = None
            self.current_confidence = 0.0
        
        # Thêm vote mới
        self.votes.append((current_time, prediction, confidence))
        
        # Xóa votes cũ (quá vote_lifetime)
        cutoff_time = current_time - self.vote_lifetime
        while self.votes and self.votes[0][0] < cutoff_time:
            self.votes.popleft()
        
        # ĐIỀU KIỆN 1: Đủ số votes đầu vào (min_votes)
        # Phải có ít nhất min_votes mới bắt đầu kiểm tra các điều kiện khác
        if len(self.votes) < self.min_votes:
            return None, 0.0
        
        # ĐIỀU KIỆN 2: Đủ thời gian giữ tay (acceptance_time)
        # Phải giữ tay trong ít nhất acceptance_time từ vote đầu tiên
        first_vote_time = self.votes[0][0]
        if (current_time - first_vote_time) < self.acceptance_time:
            return None, 0.0
        
        # ĐIỀU KIỆN 3: Đủ tỷ lệ đồng ý (vote_threshold)
        # Tính tỷ lệ dựa vào TẤT CẢ votes trong vote_lifetime (KHÔNG phải chỉ trong acceptance_time)
        # Sau khi reset, tất cả votes đều là gesture mới
        predictions = [pred for _, pred, _ in self.votes]
        counter = Counter(predictions)
        gesture, count = counter.most_common(1)[0]
        ratio = count / len(self.votes)  # Tỷ lệ = số votes gesture / tổng số votes trong vote_lifetime
        
        # Chỉ chấp nhận nếu đạt vote_threshold (80%)
        if ratio >= self.vote_threshold:
            self.current_gesture = gesture
            self.current_confidence = ratio
            return gesture, ratio
        
        # Chưa đạt threshold: trả về None để hiển thị progress
        return None, 0.0
    
    def get_progress(self):
        if len(self.votes) < self.min_votes:
            return 0.0, 0.0
        first_vote_time = self.votes[0][0]
        elapsed_time = time.time() - first_vote_time
        time_progress = min(100, (elapsed_time / self.acceptance_time) * 100)
        predictions = [pred for _, pred, _ in self.votes]
        counter = Counter(predictions)
        count = counter.most_common(1)[0][1]
        vote_progress = (count / len(self.votes)) * 100
        return vote_progress, time_progress
    
    def reset(self):
        self.votes.clear()
        self.current_gesture = None
        self.current_confidence = 0.0

# Global gesture voters (một voter cho mỗi hand)
gesture_voters = {}

# 2.1.5. Optimized Prediction Function (tf.function)
# SavedModel: gọi qua signature 'serving_default' với input dict {'input_layer_1': features_batch}
@tf.function(reduce_retracing=True)
def predict_gesture(features_batch):
    # Đảm bảo dtype là float32 (SavedModel yêu cầu)
    if isinstance(features_batch, np.ndarray):
        features_batch = tf.constant(features_batch, dtype=tf.float32)
    elif not isinstance(features_batch, tf.Tensor):
        features_batch = tf.convert_to_tensor(features_batch, dtype=tf.float32)
    
    # SavedModel format: input phải là dict với key 'input_layer_1'
    result = model_signature(input_layer_1=features_batch)
    # Output là dict {'output_0': tensor}, trả về tensor
    return result['output_0']


# 2.2. Normalize Features Function
def normalize_features(landmarks_array):
    """
    Normalize landmarks: relative + scale normalization + orientation features
    Input: landmarks_array shape (21, 2) với (x, y) [đã normalized [0,1] từ MediaPipe]
    Output: NUM_FEATURES features [x0, y0, x1, y1, ..., x20, y20, y_hand_x, y_hand_y, x_hand_x, x_hand_y] đã normalize (np.float32)
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
    
    #  ORIENTATION FEATURES (HAND LOCAL SPACE) 
    # Tính 2 trục cơ bản của bàn tay để phân biệt hướng và rotation
    # PHẢI GIỐNG HỆT với hàm normalize_features()
    # 
    # QUY TRÌNH:
    # 1. Dựng hệ trục bàn tay (hand local coordinate system):
    #    - Y_hand = wrist → mean(MCPs) (trục DỌC của bàn tay)
    #    - X_hand = index_MCP → pinky_MCP (trục NGANG của bàn tay)
    # 2. Normalize cả 2 trục thành unit vectors
    # 3. Lưu vào features [indices 42-45]
    # 
    # TẠI SAO CẦN CẢ 2 TRỤC?
    # - CHỈ Y_hand: Phân biệt Up/Down/Left/Right nhưng KHÔNG biết rotation angle
    # - CẢ Y_hand + X_hand: Phân biệt đầy đủ orientation 2D (hướng + rotation)
    # 
    # VÍ DỤ:
    # - Thumbs Up thẳng: Y=(0,-1), X=(+1,0)
    # - Thumbs Up xoay 45°: Y=(-0.7,-0.7), X=(+0.7,-0.7)
    # - Thumbs Up tay trái: Y=(0,-1), X=(-1,0)  ← Model đã train với flip augmentation
    # 
    # LƯU Ý:
    # - Model đã train với SYMMETRIC AUGMENTATION (flip ảnh)
    # - Symmetric gestures có thể có X_hand hướng bất kỳ (từ augmentation)
    # - Model tự học robust với cả 2 hướng X_hand cho symmetric gestures
    
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


# 2.2.1. Check Orientation Validity Function
def check_orientation_validity(landmarks_array):
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
    
    # Tính Y_hand và X_hand vectors (giống normalize_features)
    mcp_x_avg = np.mean(normalized_x[FINGER_MCP_INDICES])
    mcp_y_avg = np.mean(normalized_y[FINGER_MCP_INDICES])
    
    y_hand_x = mcp_x_avg
    y_hand_y = mcp_y_avg
    y_hand_mag = np.sqrt(y_hand_x**2 + y_hand_y**2)
    
    x_hand_x = normalized_x[PINKY_MCP_IDX] - normalized_x[INDEX_MCP_IDX]
    x_hand_y = normalized_y[PINKY_MCP_IDX] - normalized_y[INDEX_MCP_IDX]
    x_hand_mag = np.sqrt(x_hand_x**2 + x_hand_y**2)
    
    # Kiểm tra validity: cả 2 vectors phải có magnitude đủ lớn
    is_valid = (y_hand_mag >= ORIENTATION_MIN_MAGNITUDE and 
                x_hand_mag >= ORIENTATION_MIN_MAGNITUDE)
    
    return is_valid, y_hand_mag, x_hand_mag

# 2.2.2. Validate Handedness for Asymmetric Gestures
def validate_handedness_for_prediction(prediction_label, detected_handedness):
    # Chỉ validate cho asymmetric gestures
    if not (prediction_label.startswith(ASYMMETRIC_PREFIX_LH) or prediction_label.startswith(ASYMMETRIC_PREFIX_RH)):
        return True, None, None  # Symmetric gestures không cần validate
    
    # Xác định expected hand từ label
    if prediction_label.startswith(ASYMMETRIC_PREFIX_LH):
        expected_hand = 'Left'
    elif prediction_label.startswith(ASYMMETRIC_PREFIX_RH):
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

# 2.3. Map Prediction Label Function
def map_prediction_label(prediction, handedness_label, SYMMETRIC_GESTURES):
    # Symmetric gestures: bỏ prefix "S_"
    if prediction in SYMMETRIC_GESTURES:
        return prediction.replace(SYMMETRIC_PREFIX, "") if prediction.startswith(SYMMETRIC_PREFIX) else prediction
    
    # Asymmetric gestures: giữ nguyên prefix LH/RH để biết rõ tay trái/phải
    if prediction.startswith(ASYMMETRIC_PREFIX_RH):
        # Model predict "A_RH_FanLeft" → hiển thị "RH_FanLeft"
        return prediction.replace(ASYMMETRIC_PREFIX_RH, "RH_")
    
    elif prediction.startswith(ASYMMETRIC_PREFIX_LH):
        # Model predict "A_LH_FanLeft" → hiển thị "LH_FanLeft"
        return prediction.replace(ASYMMETRIC_PREFIX_LH, "LH_")
    
    # Fallback
    return prediction

# 2.4. MediaPipe Hand Landmarker
# Model MediaPipe (.task) dùng chung cho TOÀN project, đặt tại: Nhom17_DoAnXuLyAnhSo_HCMUTE/models/hand_landmarker.task
# Model TF SavedModel (phân loại cử chỉ) cũng đặt tại: Nhom17_DoAnXuLyAnhSo_HCMUTE/models/SavedModel/...

# project_root = .../Nhom17_DoAnXuLyAnhSo_HCMUTE
project_root = os.path.dirname(os.path.dirname(script_dir))

# Đường dẫn model MediaPipe
HAND_LANDMARKER_MODEL_PATH = os.path.join(project_root, "models", "hand_landmarker.task")
if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model MediaPipe: {HAND_LANDMARKER_MODEL_PATH}")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
base_options = BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)

# MediaPipe sử dụng 2-stage pipeline: BlazePalm (palm detector) + Hand landmark model
# Palm detector chỉ chạy khi cần (khi hand presence confidence thấp), không phải mỗi frame
# → Giúp tối ưu performance (theo Google Research blog)
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.VIDEO,  # dùng VIDEO mode cho webcam
    num_hands=NUM_HANDS,
    min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)
landmarker = HandLandmarker.create_from_options(options)

# Warm-up: chạy inference đầu tiên để khởi tạo model (giảm latency spike khi bắt đầu)
print("  → Warming up MediaPipe model...")
try:
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_frame.flags.writeable = False  # MediaPipe không cần modify image
    dummy_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_frame)
    landmarker.detect_for_video(dummy_mp_image, 0)
    print("  → Warm-up completed!")
except Exception as e:
    print(f"  → Warm-up failed (non-critical): {e}")

# EMA Smoothing State
# EMA (Exponential Moving Average) state for each hand
# Structure: {hand_idx: {'landmarks': array, 'last_seen': timestamp}}
ema_state = {}

def apply_ema_smoothing(hand_idx, current_landmarks, alpha=EMA_ALPHA):
    """
    Apply Exponential Moving Average smoothing to landmarks
    EMA formula: smoothed_t = alpha * current + (1 - alpha) * smoothed_t-1
    """
    if not ENABLE_EMA_SMOOTHING:
        return current_landmarks
    
    current_time = time.time()
    
    if hand_idx not in ema_state:
        # First time seeing this hand → initialize with current landmarks
        ema_state[hand_idx] = {
            'landmarks': current_landmarks.copy(),
            'last_seen': current_time
        }
        return current_landmarks
    
    # Apply EMA: smoothed = alpha * current + (1-alpha) * previous_smoothed
    prev_landmarks = ema_state[hand_idx]['landmarks']
    smoothed = alpha * current_landmarks + (1 - alpha) * prev_landmarks
    
    # Update state for next frame
    ema_state[hand_idx] = {
        'landmarks': smoothed,
        'last_seen': current_time
    }
    
    return smoothed

def cleanup_old_ema_state(current_hand_indices, max_age_seconds=5):
    global ema_state
    current_time = time.time()
    
    # Remove hands not in current frame AND not seen for >max_age_seconds
    ema_state = {
        idx: state for idx, state in ema_state.items() 
        if idx in current_hand_indices or (current_time - state['last_seen']) < max_age_seconds
    }

# 3. Queue & threading setup
stream_url = SOURCE
target_fps = 30.0

print("=" * 60)
print("CAMERA MODE - MediaPipe Hand Landmarker (keypoints + bbox)")
print("  → ESP32 Camera Stream")

# ESP32 stream: dùng FFMPEG backend cho MJPEG stream (tốt hơn default backend)
temp_cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
print(f"  → Kết nối ESP32 stream: {stream_url}")
print("  → Đảm bảo ESP32 đã khởi động và stream đang chạy")

if temp_cap.isOpened():
    detected_fps = temp_cap.get(cv2.CAP_PROP_FPS)
    temp_cap.release()
    if detected_fps and detected_fps > 1 and detected_fps < 240:
        target_fps = float(detected_fps)
        print(f"Detected camera FPS: {target_fps:.1f}")
    else:
        print("Detected camera FPS invalid (<=1 or >240). Using 30 FPS fallback.")
else:
    print("Warning: Unable to open camera for FPS detection. Using 30 FPS fallback.")

# Khởi tạo MQTT Publisher
try:
    init_mqtt_publisher()
    print("  → MQTT Publisher initialized")
except Exception as e:
    print(f"  MQTT Publisher init failed: {e}")
    raise  # Dừng chương trình nếu không init được MQTT

print(f"Source: {stream_url}")
print(f"Target FPS: {target_fps:.1f}")

total_start = time.time()

print("=" * 60)
print("MULTITHREADING MODE - MediaPipe Hand Landmarker")
print("  Thread 1: Frame Grabber (đọc frames từ camera)")
print("  Thread 2: Hand Landmarker (detect keypoints + bbox)")
print("  Main Thread: Display (hiển thị kết quả)")
print(f"  Frame buffer size: {FRAME_BUFFER_SIZE}")
print(f"  Detection buffer size: {DETECTION_BUFFER_SIZE}")
print("=" * 60)

frame_queue = Queue(maxsize=FRAME_BUFFER_SIZE)
display_frame_queue = Queue(maxsize=FRAME_BUFFER_SIZE)
detection_queue = Queue(maxsize=DETECTION_BUFFER_SIZE)

stop_flag = threading.Event()
queue_drop_count = 0
queue_drop_lock = threading.Lock()

# System activation state variables
system_active = False
last_activity_time = time.time()
SYSTEM_TIMEOUT_SECONDS = 20.0

def frame_grabber_thread():
    # Thread 1: Đọc frame từ ESP32 camera stream và đưa vào queue.
    global queue_drop_count
    
    # Retry logic cho ESP32 stream
    MAX_RETRIES = 5
    retry_count = 0
    
    while retry_count < MAX_RETRIES and not stop_flag.is_set():
        # Force FFMPEG backend cho MJPEG stream (tốt hơn default backend)
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            retry_count += 1
            print(f" Cannot open ESP32 stream (attempt {retry_count}/{MAX_RETRIES})")
            print(f"  → Check: ESP32 online? URL correct? WiFi stable?")
            print(f"  → Test URL in browser: {stream_url}")
            time.sleep(2)
            continue
        
        # Tối ưu cho MJPEG stream
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # 2 frames buffer cho MJPEG (cân bằng latency và stability)
        
        print(f"✓ ESP32 stream connected: {stream_url}")
        retry_count = 0  # Reset retry khi connect thành công
        
        frame_id = 0
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 30  # Cho phép 30 frames fail trước khi reconnect
        
        while not stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"Too many consecutive failures ({consecutive_failures}), reconnecting...")
                    break
                time.sleep(0.05)  # Đợi 50ms trước khi retry
                continue
            
            consecutive_failures = 0  # Reset khi đọc thành công
            frame_id += 1
            frame_time = time.time()
            
            frame_for_display = frame.copy()
            
            try:
                frame_queue.put((frame_id, frame, frame_time), timeout=0.01)
            except Full:
                with queue_drop_lock:
                    queue_drop_count += 1
            
            try:
                display_frame_queue.put((frame_id, frame_for_display, frame_time), timeout=0.01)
            except Full:
                pass
        
        # Release và retry nếu connection bị mất
        try:
            cap.release()
        except Exception:
            pass
        
        if not stop_flag.is_set():
            print("  → Reconnecting to ESP32...")
            time.sleep(1)
    
    if retry_count >= MAX_RETRIES:
        print(f"Failed to connect after {MAX_RETRIES} retries")
        print(f"  → Check ESP32: Is it running? Is URL correct?")
        print(f"  → Test in browser: {stream_url}")
    
    stop_flag.set()
    print("Thread 1 (Frame Grabber) stopped")


def hand_landmarker_thread():
    """
    Thread 2: Lấy frame từ queue, chạy MediaPipe Hand Landmarker (VIDEO mode)
    và đẩy kết quả (keypoints + handedness) sang detection_queue.
    """
    global queue_drop_count, is_paused
    
    print("  → HandLandmarker thread: MediaPipe Hand Landmarker (VIDEO mode)")
    
    frame_counter = 0
    
    while not stop_flag.is_set():
        # Check pause để giảm CPU khi pause
        if is_paused:
            time.sleep(0.1)
            continue
        
        item_retrieved = False
        try:
            frame_id, frame, frame_time = frame_queue.get(timeout=0.1)
            item_retrieved = True  # Đánh dấu đã lấy được item
            
            frame_counter += 1
            should_skip = DETECTION_SKIP_FRAMES > 0 and frame_counter % (DETECTION_SKIP_FRAMES + 1) != 0
            
            try:
                if should_skip:
                    # Skip frame nhưng vẫn cần task_done() ở finally
                    pass  # Sẽ gọi task_done() ở finally
                else:
                    # Convert BGR (OpenCV) sang RGB (MediaPipe yêu cầu)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False  # MediaPipe không cần modify image
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    ts_ms = int(frame_time * 1000)
                    t0 = time.time()
                    # Thread-safe: dùng lock để tránh race condition khi Settings recreate landmarker
                    with landmarker_lock:
                        result = landmarker.detect_for_video(mp_image, ts_ms)
                    t1 = time.time()
                    
                    inference_time = t1 - t0
                    payload = (frame_id, result, inference_time, t1)

                    try:
                        detection_queue.put(payload, timeout=0.01)
                    except Full:
                        with queue_drop_lock:
                            queue_drop_count += 1
            except Exception as e:
                print(f" Error in HandLandmarker thread processing: {e}")
            finally:
                # Đảm bảo task_done() chỉ được gọi khi đã lấy được item
                if item_retrieved:
                    frame_queue.task_done()
            
        except Empty:
            if stop_flag.is_set():
                break
            continue
        except Exception as e:
            print(f" Error in HandLandmarker thread (queue get): {e}")
            continue
    
    print("Thread 2 (HandLandmarker) stopped")


stream_url_str = str(stream_url)
print(f"Starting MediaPipe Hand Landmarker with source: {stream_url_str[:80]}{'...' if len(stream_url_str) > 80 else ''}")

thread1 = threading.Thread(target=frame_grabber_thread, daemon=True)
thread2 = threading.Thread(target=hand_landmarker_thread, daemon=True)

thread1.start()
time.sleep(0.5)
thread2.start()

pred_start = time.time()

# 4. Hiển thị real-time
total_objects = 0
frame_count = 0
MAX_FPS_HISTORY = 300
fps_list = []
frame_intervals = []
display_latencies = []
inference_fps_list = []
inference_times = []
input_fps_list = []

prev_display_time = time.time()
prev_capture_time = None

# Thread-safe shared state: latest_detection được khởi tạo None ở global scope
# Tất cả truy cập đều được bảo vệ bằng latest_detection_lock để tránh race condition
latest_detection = None
latest_detection_lock = threading.Lock()

# Thread-safe lock cho landmarker (tránh race condition khi recreate trong Settings)
landmarker_lock = threading.Lock()

# Cache container size để tránh gọi winfo_width/height mỗi frame (performance)
cached_container_size = {'w': WINDOW_WIDTH, 'h': WINDOW_HEIGHT, 'last_scale': 1.0, 'last_w': 0, 'last_h': 0}
cached_metrics_values = {}  # Cache metrics values để chỉ update khi thay đổi

# UI State
is_paused = False

# Tkinter UI Setup
try:
    root = tk.Tk()
    root.title("MediaPipe Hand Landmarker - Real-time Detection")
    
    # Tính toán kích thước window
    INFO_PANEL_WIDTH = 350
    total_width = WINDOW_WIDTH + INFO_PANEL_WIDTH + 40
    total_height = WINDOW_HEIGHT + 100
    root.geometry(f"{total_width}x{total_height}")
    root.configure(bg='#1e1e1e')
    root.minsize(800, 500)  # Kích thước tối thiểu
    
    # Căn giữa window trên màn hình khi khởi động
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - root.winfo_width()) // 2
    y = (screen_height - root.winfo_height()) // 2 - 35
    root.geometry(f"+{x}+{y}")
    
    #  HEADER 
    header_frame = tk.Frame(root, bg='#2d2d2d', height=50)
    header_frame.pack(fill=tk.X, padx=0, pady=0)
    header_frame.pack_propagate(False)
    
    title_label = tk.Label(
        header_frame,
        text="MediaPipe Hand Landmarker",
        font=('Segoe UI', 16, 'bold'),
        bg='#2d2d2d',
        fg='#ffffff'
    )
    title_label.pack(side=tk.LEFT, padx=15, pady=10)
    
    status_label = tk.Label(
        header_frame,
        text="● Ready",
        font=('Segoe UI', 10),
        bg='#2d2d2d',
        fg='#00ff00'
    )
    status_label.pack(side=tk.RIGHT, padx=15, pady=10)
    
    #  MAIN CONTENT AREA 
    main_frame = tk.Frame(root, bg='#1e1e1e')
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    #  LEFT SIDE: INFO PANEL 
    info_panel = tk.Frame(main_frame, bg='#252525', width=INFO_PANEL_WIDTH)
    info_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
    info_panel.pack_propagate(False)
    
    #  GROUP INFO SECTION 
    group_info_frame = tk.Frame(info_panel, bg='#252525')
    group_info_frame.pack(fill=tk.X, padx=15, pady=(15, 10))
    
    group_title = tk.Label(
        group_info_frame,
        text="Nhóm 17:",
        font=('Segoe UI', 14, 'bold'),
        bg='#252525',
        fg='#ffffff',
        anchor='w'
    )
    group_title.pack(anchor='w', pady=(0, 8))
    
    # Danh sách thành viên
    members = [
        "23110203 Phạm Trần Thiên Đăng",
        "23110235 Phạm Võ Nhất Kha",
        "23110280 Huỳnh Thanh Nhân",
        "23110327 Huỳnh Ngọc Thắng"
    ]
    
    for member in members:
        member_label = tk.Label(
            group_info_frame,
            text=member,
            font=('Segoe UI', 10, 'bold'),
            bg='#252525',
            fg='#ffffff',
            anchor='w'
        )
        member_label.pack(anchor='w', pady=2)
    
    # Separator line
    separator = tk.Frame(info_panel, bg='#3d3d3d', height=1)
    separator.pack(fill=tk.X, padx=15, pady=10)
    
    # Title cho info panel
    info_title = tk.Label(
        info_panel,
        text="Performance Metrics",
        font=('Segoe UI', 11, 'bold'),
        bg='#252525',
        fg='#ffffff',
        anchor='w'
    )
    info_title.pack(fill=tk.X, padx=15, pady=(0, 8))
    
    # Metrics container với vertical layout
    metrics_container = tk.Frame(info_panel, bg='#252525')
    metrics_container.pack(fill=tk.X, padx=15, pady=(0, 10))
    
    # Metrics labels (sẽ được update trong update_frame)
    metrics_labels = {}
    metric_configs = [
        ('target_fps', 'Target FPS', '#00a8ff'),
        ('latency', 'Latency', '#00ff00'),
        ('inference_time', 'Inference Time', '#ffff00'),
        ('objects', 'Objects Detected', '#ff6b6b'),
        ('input_fps', 'Input FPS', '#ffa500'),
        ('inference_fps', 'MediaPipe FPS', '#00a8ff'),
        ('display_fps', 'Display FPS', '#00ff00'),
    ]
    
    # Tạo vertical layout cho metrics (gọn hơn)
    for key, label, color in metric_configs:
        # Metric container
        metric_frame = tk.Frame(metrics_container, bg='#252525')
        metric_frame.pack(fill=tk.X, pady=3)
        
        # Label name
        name_label = tk.Label(
            metric_frame,
            text=f"{label}:",
            font=('Segoe UI', 8),
            bg='#252525',
            fg='#aaaaaa',
            anchor='w'
        )
        name_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Value label
        value_label = tk.Label(
            metric_frame,
            text="--",
            font=('Consolas', 9, 'bold'),
            bg='#252525',
            fg=color,
            anchor='w'
        )
        value_label.pack(side=tk.LEFT)
        
        metrics_labels[key] = value_label
    
    # Separator line trước console
    separator2 = tk.Frame(info_panel, bg='#3d3d3d', height=1)
    separator2.pack(fill=tk.X, padx=15, pady=(10, 8))
    
    #  CONSOLE LOG SECTION 
    console_title = tk.Label(
        info_panel,
        text="Console Log",
        font=('Segoe UI', 11, 'bold'),
        bg='#252525',
        fg='#ffffff',
        anchor='w'
    )
    console_title.pack(fill=tk.X, padx=15, pady=(0, 5))
    
    # Console text widget với scrollbar
    console_frame = tk.Frame(info_panel, bg='#252525')
    console_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
    
    console_text = scrolledtext.ScrolledText(
        console_frame,
        height=8,
        font=('Consolas', 9),
        bg='#1e1e1e',
        fg='#00ff00',
        insertbackground='#00ff00',
        wrap=tk.WORD,
        relief=tk.FLAT,
        borderwidth=0,
        padx=5,
        pady=5
    )
    console_text.pack(fill=tk.BOTH, expand=True)
    console_text.config(state=tk.DISABLED)  # Chỉ cho phép append, không cho edit
    
    # Queue để buffer messages từ các threads (thread-safe)
    console_queue = Queue()
    
    # Class để redirect stdout vào console widget (thread-safe)
    class ConsoleRedirect:
        def __init__(self, message_queue):
            self.message_queue = message_queue
            self.stdout = sys.stdout
            self.buffer = ""  # Buffer để tích lũy message từ nhiều lần write()
        
        def write(self, message):
            if not message:
                return
            
            # Tích lũy message vào buffer
            self.buffer += message
            
            # Nếu có newline trong buffer, tách ra và đưa vào queue
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                # Đưa từng dòng vào queue (mỗi dòng là một entry riêng)
                self.message_queue.put(line)
        
        def flush(self):
            # Khi flush (print() tự động gọi sau mỗi print()), đưa phần còn lại vào queue
            if self.buffer:
                cleaned = self.buffer.rstrip('\r\n')
                if cleaned:  # Chỉ đưa nếu còn nội dung sau khi strip
                    self.message_queue.put(cleaned)
                self.buffer = ""
    
    # Redirect stdout vào console queue
    console_redirect = ConsoleRedirect(console_queue)
    sys.stdout = console_redirect
    
    # Function để process console queue từ main thread
    def process_console_queue():
        try:
            # Kiểm tra widget vẫn tồn tại và root vẫn alive
            if stop_flag.is_set() or not root.winfo_exists():
                return
            
            while True:
                try:
                    message = console_queue.get_nowait()
                    try:
                        console_text.config(state=tk.NORMAL)
                        # Insert message (đã được strip newline rồi)
                        console_text.insert(tk.END, message)
                        # LUÔN LUÔN thêm 2 dòng trống để tạo khoảng cách
                        console_text.insert(tk.END, '\n\n')
                        console_text.see(tk.END)  # Auto scroll xuống cuối
                        # Giới hạn số dòng (giữ 500 dòng gần nhất)
                        lines = int(console_text.index('end-1c').split('.')[0])
                        if lines > 500:
                            console_text.delete('1.0', f'{lines-500}.0')
                        console_text.config(state=tk.DISABLED)
                    except (tk.TclError, RuntimeError):
                        # Widget đã bị destroy, restore stdout và dừng
                        try:
                            sys.stdout = sys.__stdout__
                        except:
                            pass
                        return
                except Empty:
                    break
        except Exception as e:
            # Fallback: print ra stdout gốc nếu có lỗi
            try:
                sys.__stdout__.write(f"Console error: {e}\n")
            except:
                pass
        
        # Schedule lại để check queue tiếp
        if not stop_flag.is_set() and root.winfo_exists():
            root.after(50, process_console_queue)  # Check mỗi 50ms
    
    # Bắt đầu process console queue
    root.after(100, process_console_queue)
    
    #  RIGHT SIDE: VIDEO DISPLAY 
    video_panel = tk.Frame(main_frame, bg='#1e1e1e')
    video_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    # Video container với border
    video_container = tk.Frame(video_panel, bg='#000000', relief=tk.RAISED, bd=2)
    video_container.pack(fill=tk.BOTH, expand=True)
    
    # Video label
    video_label = tk.Label(
        video_container,
        bg='#000000',
        text="Initializing camera...",
        fg='#888888',
        font=('Segoe UI', 12),
        anchor=tk.CENTER,
        justify=tk.CENTER
    )
    video_label.pack(fill=tk.BOTH, expand=True)
    
    # Callback để update cached container size khi window resize
    def update_container_cache(event=None):
        global cached_container_size
        try:
            w = video_container.winfo_width()
            h = video_container.winfo_height()
            if w > 1 and h > 1:
                cached_container_size['w'] = w
                cached_container_size['h'] = h
        except Exception:
            pass
    
    # Bind resize event để update cache
    video_container.bind('<Configure>', update_container_cache)
    root.bind('<Configure>', update_container_cache)
    
    #  KEYBOARD SHORTCUTS 
    def toggle_pause():
        global is_paused
        is_paused = not is_paused
        if status_label:
            if is_paused:
                status_label.config(text="● Paused", fg='#ffa500')
            else:
                status_label.config(text="● Running", fg='#00ff00')
    
    # Bind keyboard shortcuts
    root.bind('<space>', lambda e: toggle_pause())
    root.focus_set()  # Focus để nhận keyboard events
    
    # Handle window close
    def on_closing():
        # Restore stdout trước khi print (tránh lỗi khi widget đã bị destroy)
        try:
            sys.stdout = sys.__stdout__
        except:
            pass
        print("\nStopped by user (closed window)")
        stop_flag.set()
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    #  SETTINGS PANEL 
    settings_window = None
    
    def open_settings():
        global settings_window
        
        if settings_window is not None:
            try:
                if settings_window.winfo_exists():
                    settings_window.lift()
                    settings_window.focus()
                    return
            except Exception:
                pass
        
        settings_window = tk.Toplevel(root)
        settings_window.title("Settings")
        settings_window.geometry("600x550")
        settings_window.configure(bg='#1e1e1e')
        settings_window.resizable(False, False)
        
        # Center settings window
        settings_window.update_idletasks()
        x = (settings_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (settings_window.winfo_screenheight() // 2) - (550 // 2)
        settings_window.geometry(f"+{x}+{y}")
        
        # Header
        header = tk.Label(
            settings_window,
            text="Settings",
            font=('Segoe UI', 16, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff',
            pady=15
        )
        header.pack(fill=tk.X)
        
        # Content frame
        content_frame = tk.Frame(settings_window, bg='#1e1e1e', padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Container cho 2 cột
        columns_frame = tk.Frame(content_frame, bg='#1e1e1e')
        columns_frame.pack(fill=tk.BOTH, expand=True)
        
        #  COLUMN 1: Performance Settings 
        perf_frame = tk.LabelFrame(
            columns_frame,
            text="Performance Settings",
            font=('Segoe UI', 11, 'bold'),
            bg='#252525',
            fg='#ffffff',
            padx=15,
            pady=15
        )
        perf_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # NUM_HANDS
        tk.Label(perf_frame, text="Number of Hands:", bg='#252525', fg='#aaaaaa', anchor='w').pack(fill=tk.X, pady=5)
        num_hands_var = tk.IntVar(value=NUM_HANDS)
        num_hands_scale = tk.Scale(
            perf_frame,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            variable=num_hands_var,
            bg='#252525',
            fg='#ffffff',
            highlightbackground='#252525'
        )
        num_hands_scale.pack(fill=tk.X, pady=5)
        
        # MIN_DETECTION_CONFIDENCE
        tk.Label(perf_frame, text="Min Detection Confidence:", bg='#252525', fg='#aaaaaa', anchor='w').pack(fill=tk.X, pady=5)
        min_det_var = tk.DoubleVar(value=MIN_DETECTION_CONFIDENCE)
        min_det_scale = tk.Scale(
            perf_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=min_det_var,
            bg='#252525',
            fg='#ffffff',
            highlightbackground='#252525'
        )
        min_det_scale.pack(fill=tk.X, pady=5)
        
        # MIN_TRACKING_CONFIDENCE
        tk.Label(perf_frame, text="Min Tracking Confidence:", bg='#252525', fg='#aaaaaa', anchor='w').pack(fill=tk.X, pady=5)
        min_track_var = tk.DoubleVar(value=MIN_TRACKING_CONFIDENCE)
        min_track_scale = tk.Scale(
            perf_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=min_track_var,
            bg='#252525',
            fg='#ffffff',
            highlightbackground='#252525'
        )
        min_track_scale.pack(fill=tk.X, pady=5)
        
        # MIN_PRESENCE_CONFIDENCE
        tk.Label(perf_frame, text="Min Presence Confidence:", bg='#252525', fg='#aaaaaa', anchor='w').pack(fill=tk.X, pady=5)
        min_presence_var = tk.DoubleVar(value=MIN_PRESENCE_CONFIDENCE)
        min_presence_scale = tk.Scale(
            perf_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=min_presence_var,
            bg='#252525',
            fg='#ffffff',
            highlightbackground='#252525'
        )
        min_presence_scale.pack(fill=tk.X, pady=5)
        
        #  COLUMN 2: EMA Settings 
        ema_frame = tk.LabelFrame(
            columns_frame,
            text="EMA Smoothing",
            font=('Segoe UI', 11, 'bold'),
            bg='#252525',
            fg='#ffffff',
            padx=15,
            pady=15
        )
        ema_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # ENABLE_EMA_SMOOTHING
        ema_enable_var = tk.BooleanVar(value=ENABLE_EMA_SMOOTHING)
        tk.Checkbutton(
            ema_frame,
            text="Enable EMA Smoothing",
            variable=ema_enable_var,
            bg='#252525',
            fg='#ffffff',
            selectcolor='#1e1e1e',
            activebackground='#252525',
            activeforeground='#ffffff'
        ).pack(anchor='w', pady=5)
        
        # EMA_ALPHA
        tk.Label(ema_frame, text="EMA Alpha:", bg='#252525', fg='#aaaaaa', anchor='w').pack(fill=tk.X, pady=5)
        ema_alpha_var = tk.DoubleVar(value=EMA_ALPHA)
        ema_alpha_scale = tk.Scale(
            ema_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=ema_alpha_var,
            bg='#252525',
            fg='#ffffff',
            highlightbackground='#252525'
        )
        ema_alpha_scale.pack(fill=tk.X, pady=5)
        
        # Buttons (luôn ở bottom, không bị che)
        button_frame = tk.Frame(settings_window, bg='#1e1e1e', pady=15, padx=20)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        def apply_settings():
            """Apply settings changes"""
            global NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_PRESENCE_CONFIDENCE, MIN_TRACKING_CONFIDENCE
            global ENABLE_EMA_SMOOTHING, EMA_ALPHA, landmarker, is_paused
            
            new_num_hands = num_hands_var.get()
            new_min_det = min_det_var.get()
            new_min_track = min_track_var.get()
            new_min_presence = min_presence_var.get()
            new_ema_enable = ema_enable_var.get()
            new_ema_alpha = ema_alpha_var.get()
            
            # Áp dụng EMA settings ngay
            ENABLE_EMA_SMOOTHING = new_ema_enable
            EMA_ALPHA = new_ema_alpha
            
            # Kiểm tra xem có cần recreate landmarker không
            need_recreate = (
                NUM_HANDS != new_num_hands or
                MIN_DETECTION_CONFIDENCE != new_min_det or
                MIN_PRESENCE_CONFIDENCE != new_min_presence or
                MIN_TRACKING_CONFIDENCE != new_min_track
            )
            
            if need_recreate:
                # PAUSE detection thread trước khi recreate để tránh race condition
                old_pause_state = is_paused
                is_paused = True  # Pause để thread2 không access landmarker
                time.sleep(0.3)  # Đợi thread2 hoàn thành frame hiện tại
                
                # Update global variables
                NUM_HANDS = new_num_hands
                MIN_DETECTION_CONFIDENCE = new_min_det
                MIN_PRESENCE_CONFIDENCE = new_min_presence
                MIN_TRACKING_CONFIDENCE = new_min_track
                
                # Recreate landmarker với options mới (thread-safe)
                try:
                    # Dùng lock để đảm bảo thread2 không đang sử dụng landmarker
                    with landmarker_lock:
                        # Đóng landmarker cũ
                        if landmarker:
                            landmarker.close()
                        
                        # Tạo options mới
                        new_options = HandLandmarkerOptions(
                            base_options=base_options,
                            running_mode=VisionRunningMode.VIDEO,
                            num_hands=NUM_HANDS,
                            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
                            min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
                            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
                        )
                        
                        # Tạo landmarker mới
                        landmarker = HandLandmarker.create_from_options(new_options)
                        
                        # Warm-up landmarker mới
                        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        dummy_frame.flags.writeable = False
                        dummy_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_frame)
                        landmarker.detect_for_video(dummy_mp_image, 0)
                    
                    print(f"Landmarker recreated with new settings:")
                    print(f"  NUM_HANDS={NUM_HANDS}, MIN_DET={MIN_DETECTION_CONFIDENCE:.2f}, "
                          f"MIN_PRESENCE={MIN_PRESENCE_CONFIDENCE:.2f}, MIN_TRACK={MIN_TRACKING_CONFIDENCE:.2f}")
                    
                    # Restore pause state
                    is_paused = old_pause_state
                except Exception as e:
                    print(f"Error recreating landmarker: {e}")
                    is_paused = old_pause_state  # Restore pause state nếu có lỗi
                    return
            
            print(f"Settings applied:")
            print(f"  NUM_HANDS={NUM_HANDS}, MIN_DET={MIN_DETECTION_CONFIDENCE:.2f}, "
                  f"MIN_PRESENCE={MIN_PRESENCE_CONFIDENCE:.2f}, MIN_TRACK={MIN_TRACKING_CONFIDENCE:.2f}")
            print(f"  EMA={ENABLE_EMA_SMOOTHING}, ALPHA={EMA_ALPHA:.2f}")
        
        def close_settings():
            global settings_window
            if settings_window:
                settings_window.destroy()
            settings_window = None
        
        tk.Button(
            button_frame,
            text="Apply",
            command=apply_settings,
            bg='#00a8ff',
            fg='#ffffff',
            font=('Segoe UI', 10, 'bold'),
            padx=20,
            pady=5,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame,
            text="Close",
            command=close_settings,
            bg='#666666',
            fg='#ffffff',
            font=('Segoe UI', 10),
            padx=20,
            pady=5,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)
        
        # Handle close
        settings_window.protocol("WM_DELETE_WINDOW", close_settings)
    
    # Settings button in header
    settings_btn = tk.Button(
        header_frame,
        text="⚙ Settings",
        command=open_settings,
        bg='#2d2d2d',
        fg='#ffffff',
        font=('Segoe UI', 9),
        relief=tk.FLAT,
        padx=10,
        pady=5,
        cursor='hand2',
        activebackground='#3d3d3d',
        activeforeground='#ffffff'
    )
    settings_btn.pack(side=tk.RIGHT, padx=5)
    
    print("Tkinter UI initialized")
except Exception as e:
    raise RuntimeError(f"Không thể khởi tạo Tkinter UI: {e}") from e

current_photo = None

# Helper Functions
def limit_list_size(data_list, max_size):
    """Giới hạn kích thước list, chỉ giữ N giá trị gần nhất"""
    if len(data_list) > max_size:
        return data_list[-max_size:]
    return data_list

def fps_text(val, avg=None):
    """Format FPS text với optional average value"""
    return f"{val:.1f} (avg: {avg:.1f})" if avg is not None else f"{val:.1f}"

def ms_text(val, avg=None):
    """Format milliseconds text với optional average value"""
    return f"{val:.1f}ms (avg: {avg:.1f}ms)" if avg is not None else f"{val:.1f}ms"

def moving_avg(data_list, window=30):
    """Tính trung bình trượt (moving average)"""
    if not data_list:
        return None
    return sum(data_list[-window:]) / min(window, len(data_list))

def get_track_color(track_id):
    """Tạo màu ổn định từ track_id"""
    hash_val = hash(str(track_id)) % (256**3)
    r = max(100, (hash_val & 0xFF0000) >> 16)
    g = max(100, (hash_val & 0x00FF00) >> 8)
    b = max(100, hash_val & 0x0000FF)
    return (r, g, b)

def draw_keypoints(frame, keypoints, color=(0, 255, 255), radius=3, conf_threshold=0.3):
    """
    Vẽ keypoints lên frame
        frame: Frame để vẽ
        keypoints: numpy array shape (num_keypoints, 3) với (x, y, confidence) hoặc (num_keypoints, 2) với (x, y)
        color: Màu keypoints (BGR)
        radius: Bán kính điểm keypoint
        conf_threshold: Ngưỡng confidence tối thiểu để vẽ keypoint
    """
    if keypoints is None or len(keypoints) == 0:
        return
    
    frame_h, frame_w = frame.shape[:2]
    
    radius_outer = radius + 1
    white = (255, 255, 255)
    
    # Keypoints shape: (num_keypoints, 3) với (x, y, confidence) hoặc (num_keypoints, 2) với (x, y)
    for kp in keypoints:
        if len(kp) >= 2:
            x, y = float(kp[0]), float(kp[1])
            conf = float(kp[2]) if len(kp) > 2 else 1.0
            
            # Vẽ keypoint nếu confidence đủ và tọa độ hợp lệ (>= 0 và < frame size)
            if conf >= conf_threshold and 0 <= x < frame_w and 0 <= y < frame_h:
                x, y = int(x), int(y)
                # Vẽ điểm keypoint với viền trắng mỏng để dễ nhìn
                cv2.circle(frame, (x, y), radius_outer, white, -1)  # Viền trắng
                cv2.circle(frame, (x, y), radius, color, -1)  # Điểm keypoint

def draw_hand_skeleton(frame, keypoints, color=(0, 255, 255), thickness=1, conf_threshold=0.3):
    """
    Vẽ skeleton connections cho hand keypoints (21 keypoints cho hand)
    
    Cấu trúc 21 keypoints theo MediaPipe:
    - 0: Wrist (cổ tay)
    - 1-4: Thumb (ngón cái): 1=CMC, 2=MCP, 3=IP, 4=Tip
    - 5-8: Index (ngón trỏ): 5=MCP, 6=PIP, 7=DIP, 8=Tip
    - 9-12: Middle (ngón giữa): 9=MCP, 10=PIP, 11=DIP, 12=Tip
    - 13-16: Ring (ngón áp út): 13=MCP, 14=PIP, 15=DIP, 16=Tip
    - 17-20: Pinky (ngón út): 17=MCP, 18=PIP, 19=DIP, 20=Tip
    
    Connections này khớp với MediaPipe solutions.hands.HAND_CONNECTIONS

        frame: Frame để vẽ
        keypoints: numpy array shape (21, 3) với (x, y, confidence) hoặc (21, 2) với (x, y)
        color: Màu đường nối (BGR)
        thickness: Độ dày đường nối
        conf_threshold: Ngưỡng confidence tối thiểu để vẽ connection
    """
    if keypoints is None or len(keypoints) < 21:
        return
    
    frame_h, frame_w = frame.shape[:2]
    
    # Hand keypoint connections theo MediaPipe HAND_CONNECTIONS
    # Wrist to finger bases (CMC cho thumb, MCP cho các ngón khác)
    wrist_to_fingers = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
    
    # Thumb: CMC -> MCP -> IP -> Tip
    thumb_chain = [(1, 2), (2, 3), (3, 4)]
    
    # Index finger: MCP -> PIP -> DIP -> Tip
    index_chain = [(5, 6), (6, 7), (7, 8)]
    
    # Middle finger: MCP -> PIP -> DIP -> Tip
    middle_chain = [(9, 10), (10, 11), (11, 12)]
    
    # Ring finger: MCP -> PIP -> DIP -> Tip
    ring_chain = [(13, 14), (14, 15), (15, 16)]
    
    # Pinky finger: MCP -> PIP -> DIP -> Tip
    pinky_chain = [(17, 18), (18, 19), (19, 20)]
    
    # Tất cả connections
    all_connections = wrist_to_fingers + thumb_chain + index_chain + middle_chain + ring_chain + pinky_chain
    
    for start_idx, end_idx in all_connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            kp1 = keypoints[start_idx]
            kp2 = keypoints[end_idx]
            
            if len(kp1) >= 2 and len(kp2) >= 2:
                x1, y1 = float(kp1[0]), float(kp1[1])
                x2, y2 = float(kp2[0]), float(kp2[1])
                conf1 = float(kp1[2]) if len(kp1) > 2 else 1.0
                conf2 = float(kp2[2]) if len(kp2) > 2 else 1.0
                
                if (
                    conf1 >= conf_threshold
                    and conf2 >= conf_threshold
                    and 0 <= x1 < frame_w
                    and 0 <= y1 < frame_h
                    and 0 <= x2 < frame_w
                    and 0 <= y2 < frame_h
                ):
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    cv2.line(frame, pt1, pt2, color, thickness)

# Main Update Loop
def update_frame():
    global frame_count, total_objects, prev_display_time, prev_capture_time
    global latest_detection, current_photo
    global inference_fps_list, inference_times, input_fps_list, fps_list, frame_intervals, display_latencies
    global cached_container_size, cached_metrics_values, is_paused
    global system_active, last_activity_time
    
    if stop_flag.is_set():
        if root:
            root.quit()
        return
    
    # Skip update if paused
    if is_paused:
        if root and not stop_flag.is_set():
            root.after(200, update_frame)  # Check lại sau 200ms
        return
    
    try:
        # Lấy frame mới nhất từ display_frame_queue (skip frames cũ để giảm lag)
        frame_id, frame_original, frame_time = None, None, None
        
        try:
            # Lấy tất cả frames và chỉ giữ frame mới nhất (skip frames cũ)
            while True:
                frame_id, frame_original, frame_time = display_frame_queue.get_nowait()
                display_frame_queue.task_done()
        except Empty:
            if frame_original is None:
                try:
                    frame_id, frame_original, frame_time = display_frame_queue.get(timeout=0.01)
                    display_frame_queue.task_done()
                except Empty:
                    # Schedule next update
                    if root and not stop_flag.is_set():
                        root.after(10, update_frame)
                    return
        
        if frame_original is None:
            if root and not stop_flag.is_set():
                root.after(10, update_frame)
            return
        
        # Lấy kích thước frame từ frame_original (sau khi đã check None)
        try:
            frame_w, frame_h = frame_original.shape[1], frame_original.shape[0]
        except (AttributeError, IndexError) as e:
            print(f" Error getting frame dimensions: {e}")
            if root and not stop_flag.is_set():
                root.after(10, update_frame)
            return
        
        # Check detection_queue non-blocking để lấy hand landmarks mới nhất
        result = None
        inference_time = 0
        inference_end_time = None
        
        try:
            detection_data = detection_queue.get_nowait()
            frame_id_det, result, inference_time, inference_end_time = detection_data
            detection_queue.task_done()
            with latest_detection_lock:
                latest_detection = (result, inference_time, inference_end_time)
            # Kiểm tra inference_time > 0 trước khi tính reciprocal (tránh ZeroDivision)
            # Dùng epsilon nhỏ để tránh division by very small numbers
            if inference_time and inference_time > 1e-6:
                inference_fps_list.append(1.0 / inference_time)
                inference_fps_list = limit_list_size(inference_fps_list, MAX_FPS_HISTORY)
                inference_times.append(inference_time)
                inference_times = limit_list_size(inference_times, MAX_FPS_HISTORY)
        except Empty:
            with latest_detection_lock:
                if latest_detection is not None:
                    result, inference_time, inference_end_time = latest_detection
        
        # Đo Input FPS thực tế
        if prev_capture_time is not None:
            capture_interval = frame_time - prev_capture_time
            if capture_interval > 0:
                input_fps_list.append(1.0 / capture_interval)
                input_fps_list = limit_list_size(input_fps_list, MAX_FPS_HISTORY)
        prev_capture_time = frame_time
        
        # Tính latency (chỉ khi có inference_end_time hợp lệ)
        current_display_time = time.time()
        if inference_end_time is not None:
            display_latency = current_display_time - inference_end_time
            display_latencies.append(display_latency)
            display_latencies = limit_list_size(display_latencies, MAX_FPS_HISTORY)
        else:
            display_latency = 0  # Chưa có detection nào
        
        # Tính frame interval
        if frame_count == 0:
            frame_interval = 0
            prev_display_time = current_display_time
        else:
            frame_interval = current_display_time - prev_display_time
            prev_display_time = current_display_time
        
        if frame_interval > 0:
            frame_intervals.append(frame_interval)
            frame_intervals = limit_list_size(frame_intervals, MAX_FPS_HISTORY)
        
        frame_count += 1
        
        # KIỂM TRA TIMEOUT ĐỂ TẮT CHẾ ĐỘ NHẬN LỆNH
        if system_active and (time.time() - last_activity_time) > SYSTEM_TIMEOUT_SECONDS:
            system_active = False
            print(f"SYSTEM DEACTIVATED due to {SYSTEM_TIMEOUT_SECONDS}s inactivity")

        # Số bàn tay (dựa trên MediaPipe)
        num_objects = 0
        if result and result.hand_landmarks:
            num_objects = len(result.hand_landmarks)
        total_objects += num_objects
        
        # Tính FPS hiển thị
        current_fps = None
        if frame_interval > 0:
            current_fps = 1.0 / frame_interval
            fps_list.append(current_fps)
            fps_list = limit_list_size(fps_list, MAX_FPS_HISTORY)
        
        # Tính trung bình các FPS metrics
        # Dùng epsilon nhỏ để tránh division by very small numbers
        current_inference_fps = (1.0 / inference_time) if (inference_time and inference_time > 1e-6) else None
        avg_fps_display = moving_avg(fps_list)
        avg_inference_fps_display = moving_avg(inference_fps_list)
        avg_input_fps_display = moving_avg(input_fps_list)
        avg_display_latency = moving_avg(display_latencies)
        avg_inference_time = moving_avg(inference_times)
        current_input_fps = input_fps_list[-1] if input_fps_list else None
        
        # Visualization (MediaPipe hand landmarks + bounding box)
        annotated_frame = frame_original.copy()
        
        # Hiển thị trạng thái hệ thống (Active/Idle)
        status_text = "SYSTEM: ACTIVE" if system_active else "SYSTEM: IDLE (Wait for 'Start')"
        status_color = (0, 255, 0) if system_active else (0, 0, 255) # Green if Active, Red if Idle
        cv2.putText(annotated_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        #  XỬ LÝ KHI KHÔNG CÓ HAND 
        has_hand = result and result.hand_landmarks and len(result.hand_landmarks) > 0
        
        if not has_hand:
            # Reset và xóa tất cả voters khi không có tay
            for hand_idx in list(gesture_voters.keys()):
                gesture_voters[hand_idx].reset()
                del gesture_voters[hand_idx]
        
        if result and result.hand_landmarks:
            try:
                # Cleanup old EMA state (prevent memory leak) - định kỳ mỗi 30 frames
                if frame_count % 30 == 0:
                    current_hand_indices = set(range(len(result.hand_landmarks)))
                    cleanup_old_ema_state(current_hand_indices, max_age_seconds=2)
                
                for hand_idx, landmarks in enumerate(result.hand_landmarks):
                    # Validate số lượng landmarks
                    if len(landmarks) != NUM_LANDMARKS:
                        continue
                    
                    # landmarks: list 21 điểm, mỗi điểm có x, y (normalized [0,1] từ MediaPipe)
                    # Validate và clamp x, y trong khoảng [0, 1] để tránh crash nếu MediaPipe trả về giá trị lỗi
                    landmarks_normalized = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
                    landmarks_normalized = np.clip(landmarks_normalized, 0.0, 1.0)
                    
                    # Apply EMA smoothing to reduce jitter (trên normalized coordinates [0,1])
                    landmarks_normalized = apply_ema_smoothing(hand_idx, landmarks_normalized, alpha=EMA_ALPHA)
                    
                    # Convert sang pixel coordinates để vẽ
                    landmarks_array = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                    landmarks_array[:, 0] = landmarks_normalized[:, 0] * frame_w
                    landmarks_array[:, 1] = landmarks_normalized[:, 1] * frame_h
                    landmarks_array[:, 2] = 1.0
                    
                    xs = landmarks_array[:, 0]
                    ys = landmarks_array[:, 1]
                    
                    
                    # Bounding box theo keypoints
                    min_x, max_x = int(xs.min()), int(xs.max())
                    min_y, max_y = int(ys.min()), int(ys.max())
                
                    # Lọc theo kích thước box
                    box_w = max_x - min_x
                    box_h = max_y - min_y
                    if box_w <= 0 or box_h <= 0:
                        continue
                    box_area = box_w * box_h
                    frame_area = float(frame_w * frame_h)
                    area_ratio = box_area / frame_area if frame_area > 0 else 0.0

                    # Bỏ box quá nhỏ (nhiễu) hoặc quá lớn (thường là gần camera)
                    if area_ratio < HAND_MIN_AREA_RATIO or area_ratio > HAND_MAX_AREA_RATIO:
                        continue

                    # Lọc thêm theo độ tin cậy handedness để tránh patch mờ mờ bị gán tay
                    handedness_label = "Hand"
                    handedness_label_clean = None  # Lưu clean label để dùng cho validation sau này
                    handedness_score = 1.0
                    if result.handedness and len(result.handedness) > hand_idx:
                        entry = result.handedness[hand_idx]
                        # Xử lý an toàn: entry có thể là list/tuple hoặc object trực tiếp
                        if isinstance(entry, (list, tuple)) and len(entry) > 0:
                            cat = entry[0]
                        else:
                            cat = entry
                        
                        # Đọc category_name và score (hỗ trợ nhiều version MediaPipe)
                        name = getattr(cat, "category_name", None) or getattr(cat, "label", None) or "Hand"
                        score = getattr(cat, "score", None) or getattr(cat, "confidence", None) or 1.0
                        handedness_label = f"{name}:{float(score):.2f}"
                        handedness_label_clean = name  # Lưu clean label (chỉ "Left" hoặc "Right")
                        handedness_score = float(score)
                    # Nếu độ tin cậy handedness quá thấp thì bỏ qua (không vẽ tay)
                    # Loại bỏ các detection không chắc chắn (có thể là false positive)
                    if handedness_score < HANDEDNESS_SCORE_THRESHOLD:
                        continue
                    
                    #  ~~~~~~~~~~~~~~~~~~~~~~ GESTURE CLASSIFICATION ~~~~~~~~~~~~~~~~~~~~~~
                    # Chỉ predict gesture khi bounding box đã hợp lệ
                    
                    # FILTER: Kiểm tra orientation validity TRƯỚC khi predict
                    # Reject nếu orientation không ổn định (magnitude quá nhỏ)
                    orientation_valid, y_hand_mag, x_hand_mag = check_orientation_validity(landmarks_normalized)
                    if not orientation_valid:
                        # Orientation không ổn định → không đáng tin, bỏ qua prediction
                        if hand_idx in gesture_voters:
                            gesture_voters[hand_idx].reset()  # Reset voter để tránh tích lũy votes sai
                        gesture_display = f"Unstable Orientation (Y:{y_hand_mag:.3f}, X:{x_hand_mag:.3f})"
                        vote_confidence = 0.0
                        # Vẽ bounding box nhưng không hiển thị gesture
                        # (sẽ được xử lý ở phần vẽ bên dưới)
                    else:
                        # Extract NUM_FEATURES features từ normalized landmarks [0,1] (42 landmarks + 2 Y_hand + 2 X_hand)
                        features = normalize_features(landmarks_normalized).astype(np.float32)
                        
                        # Predict gesture
                        features_reshaped = features.reshape(1, NUM_FEATURES).astype(np.float32)
                        # SavedModel đã được compiled khi save, không cần compile lại
                        # tf.function compile function thành graph → nhanh hơn rất nhiều
                        predictions = predict_gesture(features_reshaped)
                        # Convert TensorFlow tensor về numpy
                        predictions_np = predictions.numpy() if hasattr(predictions, 'numpy') else np.array(predictions)
                        probs = predictions_np[0]  # Probability distribution
                        
                        # Lấy top-1 prediction
                        top1_idx = int(np.argmax(probs))
                        confidence = float(probs[top1_idx])
                        class_id = top1_idx
                        
                        # Tính entropy của probability distribution
                        # Entropy cao = phân phối phẳng = không chắc → reject
                        # Entropy thấp = phân phối tập trung = chắc → accept
                        EPSILON = 1e-10  # Tránh log(0)
                        entropy = -np.sum(probs * np.log(probs + EPSILON))
                        
                        # Kiểm tra class_id hợp lệ
                        if class_id not in label_mapping:
                            print(f"Warning: class_id {class_id} không có trong label_mapping, bỏ qua prediction")
                            continue
                        raw_label = label_mapping[class_id]
                        
                        # FILTER 0: Kiểm tra entropy (tránh phân phối phẳng = không chắc)
                        if entropy > ENTROPY_THRESHOLD:
                            # Entropy cao = nhiều class có xác suất gần nhau = không chắc → reject
                            if hand_idx in gesture_voters:
                                gesture_voters[hand_idx].reset()
                            gesture_display = f"High Entropy ({entropy:.2f}, uncertain)"
                            vote_confidence = 0.0
                        # FILTER 1: Reject cứng nếu confidence quá thấp → Unknown
                        elif confidence < GESTURE_REJECT_THRESHOLD:
                            if hand_idx in gesture_voters:
                                gesture_voters[hand_idx].reset()
                            gesture_display = f"Unknown (conf:{confidence:.2f})"
                            vote_confidence = 0.0
                        else:
                            # Tất cả filters đều pass → prediction hợp lệ, xử lý với GestureVoter
                            if hand_idx not in gesture_voters:
                                gesture_voters[hand_idx] = GestureVoter(
                                    acceptance_time=GESTURE_VOTER_ACCEPTANCE_TIME,
                                    vote_lifetime=GESTURE_VOTER_VOTE_LIFETIME,
                                    vote_threshold=GESTURE_VOTER_VOTE_THRESHOLD,
                                    min_votes=GESTURE_VOTER_MIN_VOTES
                                )
                            
                            # FILTER: Bỏ qua S_Nothing khi có hand (false positive)
                            # S_Nothing chỉ hợp lệ khi không có hand (đã xử lý ở trên)
                            if raw_label == 'S_Nothing':
                                # Model predict Nothing nhưng có hand → false positive, bỏ qua
                                gesture_voters[hand_idx].reset()  # Reset voter để tránh tích lũy votes sai
                                gesture_display = "Unknown"
                                vote_confidence = 0.0
                            else:
                                final_label, vote_ratio = gesture_voters[hand_idx].vote(raw_label, confidence)
                            
                            # Xử lý kết quả và map label
                            if final_label is None:
                                # Chưa đạt threshold → hiển thị progress
                                vote_progress, time_progress = gesture_voters[hand_idx].get_progress()
                                gesture_display = f"Processing... ({vote_progress:.0f}% votes, {time_progress:.0f}% time)"
                                vote_confidence = 0.0
                            else:
                                # Đã đạt threshold → validate handedness cho asymmetric gestures
                                # (handedness_label_clean đã được lấy ở trên)
                                
                                # FILTER: Validate handedness cho asymmetric gestures
                                is_valid_handedness, expected_hand, mismatch_reason = validate_handedness_for_prediction(
                                    final_label, handedness_label_clean
                                )
                                
                                if not is_valid_handedness:
                                    # Handedness không khớp → reject prediction này
                                    if hand_idx in gesture_voters:
                                        gesture_voters[hand_idx].reset()  # Reset voter để tránh tích lũy votes sai
                                    gesture_display = f"Handedness Mismatch ({mismatch_reason})"
                                    vote_confidence = 0.0
                                else:
                                    # Handedness hợp lệ → map label và hiển thị
                                    gesture_display = map_prediction_label(final_label, handedness_label_clean, SYMMETRIC_GESTURES)
                                    vote_confidence = vote_ratio * 100
                                    
                                    # Publish gesture qua MQTT
                                    try:
                                        # LỆNH THỰC TẾ: Bỏ các tiền tố A_RH_, A_LH_, S_, RH_, LH_ để ESP32 hiểu
                                        clean_command = final_label.replace(ASYMMETRIC_PREFIX_RH, "").replace(ASYMMETRIC_PREFIX_LH, "").replace(SYMMETRIC_PREFIX, "")
                                        
                                        # Logic Kích hoạt/Tắt hệ thống theo nhãn Start
                                        if clean_command == "Start":
                                            if not system_active:
                                                system_active = True
                                                print("SYSTEM ACTIVATED by Start gesture")
                                                # Gửi lệnh Start sang ESP32 để báo hiệu (còi kêu)
                                                # Dùng force=True vì sau 7s timeout, last_gesture của MQTT vẫn là 'Start'
                                                publish_gesture_command(
                                                    gesture_label=clean_command,
                                                    confidence=vote_confidence,
                                                    hand_id=hand_idx,
                                                    handedness=handedness_label_clean,
                                                    force=True
                                                )
                                            last_activity_time = time.time() # Cập nhật thời gian khi thấy Start
                                        
                                        elif system_active:
                                            # Nếu đang ở chế độ nhận lệnh, gửi tất cả các lệnh khác Start
                                            # Tránh gửi Start liên tục khi đã active
                                            publish_gesture_command(
                                                gesture_label=clean_command,
                                                confidence=vote_confidence,
                                                hand_id=hand_idx,
                                                handedness=handedness_label_clean
                                            )
                                            last_activity_time = time.time() # Cập nhật thời gian khi có lệnh hợp lệ
                                            
                                    except Exception as e:
                                        print(f"MQTT Publish error: {e}")
                    # ~~~~~~~~~~~~~~~~~~~~~

                    color = get_track_color(hand_idx)  # dùng index tay làm ID tạm

                    # Label với gesture classification
                    label = f"ID:{hand_idx} {handedness_label} | Gesture: {gesture_display} ({vote_confidence:.1f}%)"
                    
                    # Vẽ bounding box
                    cv2.rectangle(annotated_frame, (min_x, min_y), (max_x, max_y), color, 2)
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    # Đảm bảo text box không vẽ ra ngoài frame (y >= 0)
                    text_y_start = max(0, min_y - text_height - baseline - 3)
                    text_y_end = min_y
                    cv2.rectangle(
                        annotated_frame,
                        (min_x, text_y_start),
                        (min_x + text_width, text_y_end),
                        color,
                        -1,
                    )
                    # Đảm bảo text luôn visible (tránh edge case khi min_y rất nhỏ)
                    text_y = max(text_height + baseline + 2, min_y - baseline - 1)
                    white = (255, 255, 255)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (min_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        white,
                        1,
                    )
                    
                    # Skeleton + keypoints
                    draw_hand_skeleton(annotated_frame, landmarks_array, color, 1, conf_threshold=0.0)
                    draw_keypoints(annotated_frame, landmarks_array, color, 3, conf_threshold=0.0)

            except Exception as e:
                print(f" Error drawing MediaPipe results: {e}")
            
            # Cleanup voters cho hands không còn xuất hiện (xóa khỏi dict để tránh memory leak)
            # Đảm bảo cleanup ngay cả khi có exception trong quá trình xử lý
            if result and result.hand_landmarks:
                current_hand_indices = set(range(len(result.hand_landmarks)))
                for hand_idx in list(gesture_voters.keys()):
                    if hand_idx not in current_hand_indices:
                        gesture_voters[hand_idx].reset()
                        del gesture_voters[hand_idx]
        
        # Hiển thị với Tkinter
        try:
            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Resize để fill video container (scale và crop center để không có màu đen)
            h, w = rgb_frame.shape[:2]
            try:
                # Lấy kích thước video container từ cache (tránh gọi winfo mỗi frame)
                container_w = cached_container_size.get('w', WINDOW_WIDTH)
                container_h = cached_container_size.get('h', WINDOW_HEIGHT)
                
                # Nếu container chưa được render, dùng default size
                if container_w <= 1 or container_h <= 1:
                    container_w = WINDOW_WIDTH
                    container_h = WINDOW_HEIGHT
                
                # Cache resize parameters để tránh tính lại mỗi frame
                cache_key = f"{w}_{h}_{container_w}_{container_h}"
                if ('last_resize_key' not in cached_container_size or 
                    cached_container_size['last_resize_key'] != cache_key):
                    # Tính lại resize parameters khi size thay đổi
                    scale_w = container_w / w
                    scale_h = container_h / h
                    scale = max(scale_w, scale_h)
                    
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # Cache lại để dùng cho frame tiếp theo
                    cached_container_size['last_resize_key'] = cache_key
                    cached_container_size['cached_scale'] = scale
                    cached_container_size['cached_new_w'] = new_w
                    cached_container_size['cached_new_h'] = new_h
                else:
                    # Dùng lại cached values khi size không đổi
                    scale = cached_container_size['cached_scale']
                    new_w = cached_container_size['cached_new_w']
                    new_h = cached_container_size['cached_new_h']
                
                # Resize với cached parameters
                if abs(scale - 1.0) > 0.01:
                    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                    rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=interpolation)
                
                # Crop center để fit container (với validation để tránh crash)
                if new_w > container_w or new_h > container_h:
                    start_x = max(0, min((new_w - container_w) // 2, new_w - container_w))
                    start_y = max(0, min((new_h - container_h) // 2, new_h - container_h))
                    end_x = min(new_w, start_x + container_w)
                    end_y = min(new_h, start_y + container_h)
                    rgb_frame = rgb_frame[start_y:end_y, start_x:end_x]
                elif new_w < container_w or new_h < container_h:
                    # Pad với màu đen nếu nhỏ hơn container (ít khi xảy ra)
                    pad_w = (container_w - new_w) // 2
                    pad_h = (container_h - new_h) // 2
                    rgb_frame = cv2.copyMakeBorder(
                        rgb_frame, pad_h, container_h - new_h - pad_h,
                        pad_w, container_w - new_w - pad_w,
                        cv2.BORDER_CONSTANT, value=[0, 0, 0]
                    )
            except Exception:
                # Fallback: giữ nguyên kích thước
                pass
            
            # Reuse PhotoImage object để tối ưu performance (PIL.ImageTk.PhotoImage có paste() method)
            pil_image = Image.fromarray(rgb_frame)
            
            if not hasattr(video_label, 'photo_image') or video_label.photo_image is None:
                # Lần đầu: tạo mới PhotoImage
                video_label.photo_image = ImageTk.PhotoImage(image=pil_image)
                video_label.photo_image_size = pil_image.size
                video_label.config(image=video_label.photo_image, text="")
            else:
                # Update PhotoImage hiện có nếu size giống nhau (dùng paste() - nhanh hơn)
                try:
                    if hasattr(video_label, 'photo_image_size') and pil_image.size == video_label.photo_image_size:
                        # Dùng paste() để update image (tự động reflect, không cần recreate)
                        video_label.photo_image.paste(pil_image)
                    else:
                        # Tạo mới nếu size thay đổi - release PhotoImage cũ trước
                        if hasattr(video_label, 'photo_image') and video_label.photo_image is not None:
                            try:
                                del video_label.photo_image  # Release memory
                            except Exception:
                                pass
                        video_label.photo_image = ImageTk.PhotoImage(image=pil_image)
                        video_label.photo_image_size = pil_image.size
                        video_label.config(image=video_label.photo_image, text="")
                except Exception:
                    # Fallback: tạo mới PhotoImage nếu có lỗi - release PhotoImage cũ trước
                    if hasattr(video_label, 'photo_image') and video_label.photo_image is not None:
                        try:
                            del video_label.photo_image  # Release memory
                        except Exception:
                            pass
                    video_label.photo_image = ImageTk.PhotoImage(image=pil_image)
                    video_label.photo_image_size = pil_image.size
                    video_label.config(image=video_label.photo_image, text="")
            
            # Update status (không override nếu đang pause)
            if status_label and not is_paused:
                if current_fps is not None and avg_fps_display is not None:
                    status_label.config(text="● Running", fg='#00ff00')
            
            # Update metrics labels (chỉ update khi giá trị thay đổi)
            if metrics_labels:
                if current_fps is not None and avg_fps_display is not None:
                    # Tính toán các giá trị mới
                    new_values = {
                        'target_fps': f"{target_fps:.1f}",
                        'display_fps': fps_text(current_fps, avg_fps_display),
                        'inference_fps': fps_text(current_inference_fps, avg_inference_fps_display) if current_inference_fps else '--',
                        'input_fps': fps_text(current_input_fps, avg_input_fps_display) if current_input_fps else '--',
                        'latency': ms_text(display_latency*1000, avg_display_latency*1000 if avg_display_latency else None),
                        'inference_time': ms_text(inference_time*1000, avg_inference_time*1000 if avg_inference_time else None),
                        'objects': f"{num_objects}"
                    }
                    
                    # Chỉ update labels khi giá trị thay đổi
                    for key, new_value in new_values.items():
                        if key not in cached_metrics_values or cached_metrics_values[key] != new_value:
                            metrics_labels[key].config(text=new_value)
                            cached_metrics_values[key] = new_value
                else:
                    # Chưa có FPS data (chưa khởi động xong)
                    init_values = {
                        'target_fps': f"{target_fps:.1f}",
                        'display_fps': '--',
                        'inference_fps': '--',
                        'input_fps': '--',
                        'latency': '--',
                        'inference_time': '--',
                        'objects': f"{num_objects}"
                    }
                    
                    # Chỉ update labels khi giá trị thay đổi
                    for key, new_value in init_values.items():
                        if key not in cached_metrics_values or cached_metrics_values[key] != new_value:
                            metrics_labels[key].config(text=new_value)
                            cached_metrics_values[key] = new_value
        except Exception as e:
            print(f" Error updating Tkinter UI: {e}")
        
        # Print info (thống kê FPS / latency)
        if frame_count % PRINT_EVERY_N_FRAMES == 0 or frame_count <= 5:
            if len(fps_list) > 0:
                avg_frame_interval = (moving_avg(frame_intervals) or 0) * 1000
                avg_display_latency = (moving_avg(display_latencies) or 0) * 1000
                avg_fps_print = avg_fps_display or moving_avg(fps_list) or 0
                avg_inference_fps_print = moving_avg(inference_fps_list) or 0
                avg_input_fps_print = moving_avg(input_fps_list) or 0
                print(
                    f"  → Average Display FPS: {avg_fps_print:.1f} | "
                    f"Average MediaPipe FPS: {avg_inference_fps_print:.1f} | "
                    f"Average Input FPS: {avg_input_fps_print:.1f} | "
                    f"Target FPS: {target_fps:.1f} | "
                    f"Frame interval: {avg_frame_interval:.1f}ms | "
                    f"Display latency: {avg_display_latency:.1f}ms | "
                    f"Inference: {inference_time*1000:.1f}ms"
                )
        
        # Schedule next update
        if not stop_flag.is_set():
            delay = 10  # 10ms delay
            root.after(delay, update_frame)
        
    except Exception as e:
        print(f" Error in update_frame: {e}")
        if not stop_flag.is_set():
            root.after(10, update_frame)

# Chạy Tkinter main loop
root.after(10, update_frame)
root.mainloop()

# 5. Cleanup & Summary
# Dừng tất cả threads
stop_flag.set()

# Queue cleanup: dùng get_nowait() với Empty exception
try:
    while True:
        try:
            frame_queue.get_nowait()
            frame_queue.task_done()
        except Empty:
            break
    while True:
        try:
            display_frame_queue.get_nowait()
            display_frame_queue.task_done()
        except Empty:
            break
    while True:
        try:
            detection_queue.get_nowait()
            detection_queue.task_done()
        except Empty:
            break
except Exception:
    pass

# Đợi threads kết thúc hoàn toàn
if thread1.is_alive():
    thread1.join(timeout=3)
if thread2.is_alive():
    thread2.join(timeout=3)

# Đóng landmarker để giải phóng tài nguyên
try:
    landmarker.close()
except Exception:
    pass

# Disconnect MQTT Publisher
try:
    mqtt_pub = get_mqtt_publisher()
    if mqtt_pub:
        mqtt_pub.disconnect()
except Exception as e:
    print(f" MQTT Disconnect error: {e}")

# Cleanup Tkinter (nếu chưa được destroy)
try:
    if root.winfo_exists():
        root.quit()
        root.destroy()
except Exception:
    pass

pred_end = time.time()
pred_time = pred_end - pred_start

# Tính toán thống kê cuối cùng
avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
avg_frame_interval = sum(frame_intervals) / len(frame_intervals) * 1000 if frame_intervals else 0
avg_display_latency = sum(display_latencies) / len(display_latencies) * 1000 if display_latencies else 0
min_display_latency = min(display_latencies) * 1000 if display_latencies else 0
max_display_latency = max(display_latencies) * 1000 if display_latencies else 0
avg_inference_fps = sum(inference_fps_list) / len(inference_fps_list) if inference_fps_list else 0
avg_input_fps = sum(input_fps_list) / len(input_fps_list) if input_fps_list else 0

total_end = time.time()

# Restore stdout về gốc để print summary ra terminal (tránh lỗi khi widget đã bị destroy)
try:
    sys.stdout = sys.__stdout__
except:
    pass

print(f"\n{'='*60}")
print(f"REALTIME SUMMARY - MEDIAPIPE HAND LANDMARKER:")
print(f"  Backend: MediaPipe (hand_landmarker.task)")
print(f"  Total frames processed: {frame_count}")
print(f"  Total objects detected: {total_objects}")
print(f"  Target FPS: {target_fps:.1f}")
print(f"  Average Display FPS: {avg_fps:.2f}")
print(f"  Average MediaPipe FPS: {avg_inference_fps:.2f}")
print(f"  Average Input FPS: {avg_input_fps:.2f}")
print(f"  Average frame interval: {avg_frame_interval:.2f}ms")
print(f"  Average display latency: {avg_display_latency:.2f}ms")
print(f"  Min display latency: {min_display_latency:.2f}ms | Max display latency: {max_display_latency:.2f}ms")
print(f"  Total inference time: {pred_time:.2f}s")
print(f"  Total script time: {total_end - total_start:.2f} seconds")
with queue_drop_lock:
    print(f"  Queue drops (frames): {queue_drop_count}")
if avg_fps > 0:
    efficiency = (avg_fps / target_fps) * 100
    print(f"  Efficiency: {efficiency:.1f}% (vs target FPS)")