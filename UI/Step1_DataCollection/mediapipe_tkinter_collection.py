import os
import re
import time
import cv2
import warnings
import threading
from queue import Queue, Empty, Full
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, scrolledtext
import sys

import mediapipe as mp
from mediapipe.tasks.python import vision

# Giảm warning log
warnings.filterwarnings("ignore", category=UserWarning)

# 1. CONFIGURATION
SOURCE = 0  # 0 = webcam mặc định

# Performance settings
NUM_HANDS = 1  # 2 người (mỗi người 2 tay) - có thể giảm xuống 2 nếu chỉ cần 1 người
MIN_DETECTION_CONFIDENCE = 0.6  # Ngưỡng cho palm detector (BlazePalm)
MIN_PRESENCE_CONFIDENCE = 0.5   # Ngưỡng để trigger re-detection (thấp hơn = re-detect thường xuyên hơn)
MIN_TRACKING_CONFIDENCE = 0.5   # Ngưỡng cho hand tracking (landmark model)

# Filtering thresholds
HAND_MIN_AREA_RATIO = 0.0025   # ~0.25% diện tích frame (bỏ box quá nhỏ)
HAND_MAX_AREA_RATIO = 0.35     # ~35% diện tích frame (bỏ box quá lớn)
HANDEDNESS_SCORE_THRESHOLD = 0.6  # Ngưỡng confidence tối thiểu cho handedness

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

# 2. MediaPipe Hand Landmarker
script_dir = os.path.dirname(os.path.abspath(__file__))

# Gốc project: .../Nhom17_DoAnXuLyAnhSo_HCMUTE
project_root = os.path.dirname(os.path.dirname(script_dir))

# Model MediaPipe (.task) dùng chung cho TOÀN project, đặt tại: Nhom17_DoAnXuLyAnhSo_HCMUTE/models/hand_landmarker.task
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
    if not ENABLE_EMA_SMOOTHING:
        return current_landmarks
    
    current_time = time.time()
    
    if hand_idx not in ema_state:
        ema_state[hand_idx] = {
            'landmarks': current_landmarks.copy(),
            'last_seen': current_time
        }
        return current_landmarks
    
    prev_landmarks = ema_state[hand_idx]['landmarks']
    smoothed = alpha * current_landmarks + (1 - alpha) * prev_landmarks
    
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

# Tối ưu: Dùng MSMF backend trên Windows
try:
    temp_cap = cv2.VideoCapture(stream_url, cv2.CAP_MSMF)
except Exception:
    temp_cap = cv2.VideoCapture(stream_url)

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

def frame_grabber_thread():
    global queue_drop_count
    try:
        cap = cv2.VideoCapture(stream_url, cv2.CAP_MSMF)  # Windows: MSMF backend
    except Exception:
        cap = cv2.VideoCapture(stream_url)  # Fallback
    
    if not cap.isOpened():
        print(" Error: Cannot open video source")
        stop_flag.set()
        return
    
    # Tối ưu OpenCV settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer để giảm latency
    cap.set(cv2.CAP_PROP_FPS, target_fps)  # Set FPS nếu camera support
    
    frame_id = 0
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print(" End of stream or error reading frame")
            break
        
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
    
    try:
        cap.release()
    except Exception:
        pass
    stop_flag.set()
    print("Thread 1 (Frame Grabber) stopped")


def hand_landmarker_thread():
    global queue_drop_count, is_paused
    
    print("  → HandLandmarker thread: MediaPipe Hand Landmarker (VIDEO mode)")
    
    frame_counter = 0
    
    while not stop_flag.is_set():
        # Check pause để giảm CPU khi pause
        if is_paused:
            time.sleep(0.1)
            continue
        
        try:
            frame_id, frame, frame_time = frame_queue.get(timeout=0.1)
            
            frame_counter += 1
            should_skip = DETECTION_SKIP_FRAMES > 0 and frame_counter % (DETECTION_SKIP_FRAMES + 1) != 0
            
            try:
                if should_skip:
                    # Skip frame nhưng vẫn cần task_done() ở finally
                    continue
                # Convert BGR (OpenCV) sang RGB (MediaPipe yêu cầu)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False  # MediaPipe không cần modify image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                ts_ms = int(frame_time * 1000)
                t0 = time.time()
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
                print(f"Error in HandLandmarker thread processing: {e}")
            finally:
                # Đảm bảo task_done() chỉ được gọi 1 lần cho mỗi frame
                frame_queue.task_done()
            
        except Empty:
            if stop_flag.is_set():
                break
            continue
        except Exception as e:
            print(f"Error in HandLandmarker thread (queue get): {e}")
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

# Thread-safe shared state: current_frame để chụp ảnh ngẫu nhiên
current_frame = None
current_frame_lock = threading.Lock()

# Cache container size để tránh gọi winfo_width/height mỗi frame (performance)
cached_container_size = {'w': WINDOW_WIDTH, 'h': WINDOW_HEIGHT, 'last_scale': 1.0, 'last_w': 0, 'last_h': 0}
cached_metrics_values = {}  # Cache metrics values để chỉ update khi thay đổi

# UI State
is_paused = False

# Auto-save hand images settings
SAVE_HAND_IMAGES = False  # Bật/tắt tự động lưu ảnh
SAVE_DIR = os.path.join(script_dir, "dataset", "S_Test")  # Thư mục lưu ảnh
save_image_counter = 0  # Đếm số ảnh đã lưu
last_save_time = 0  # Thời gian lưu ảnh cuối cùng (để tránh lưu quá nhiều)
SAVE_INTERVAL = 0.5  # Khoảng thời gian tối thiểu giữa các lần lưu (giây)
notification_timer = None  # Timer để ẩn thông báo
notification_label = None  # Label hiển thị thông báo (sẽ được khởi tạo trong UI)
root = None  # Tkinter root window (sẽ được khởi tạo trong UI)

# Continuous capture state
auto_capture_enabled = False
auto_capture_job = None

def scan_missing_image_numbers(save_dir):
    if not os.path.exists(save_dir):
        return [], 0
    
    try:
        # Lấy danh sách tất cả các file trong thư mục
        files = os.listdir(save_dir)
        
        # Tìm tất cả các số counter từ các file có format: NUMBER.jpg
        pattern = re.compile(r"^(\d+)\.jpg$", re.IGNORECASE)
        existing_numbers = set()
        max_counter = -1
        
        for filename in files:
            match = pattern.match(filename)
            if match:
                counter = int(match.group(1))
                existing_numbers.add(counter)
                if counter > max_counter:
                    max_counter = counter
        
        # Nếu không có file nào, trả về list rỗng và bắt đầu từ 0
        if max_counter < 0:
            return [], 0
        
        # Tìm các số còn thiếu trong khoảng từ 0 đến max_counter
        missing_numbers = []
        for i in range(max_counter + 1):
            if i not in existing_numbers:
                missing_numbers.append(i)
        
        # Sắp xếp các số còn thiếu để dùng theo thứ tự
        missing_numbers.sort()
        
        # Số tiếp theo để lưu là max_counter + 1
        next_counter = max_counter + 1
        
        return missing_numbers, next_counter
    except Exception as e:
        print(f" Lỗi khi quét thư mục lưu ảnh: {e}")
        return [], 0

# Khởi tạo: quét thư mục và lưu các số còn thiếu
missing_image_numbers = []  # Queue các số còn thiếu cần lấp vào
save_image_counter = 0  # Số tiếp theo để lưu (sau khi đã lấp hết số còn thiếu)

# Quét thư mục khi khởi động
missing_image_numbers, save_image_counter = scan_missing_image_numbers(SAVE_DIR)
if missing_image_numbers:
    print(f"Đã tìm thấy {len(missing_image_numbers)} số còn thiếu: {missing_image_numbers}")
    print(f"Sẽ ưu tiên lấp vào các số này trước khi tiếp tục từ số {save_image_counter}")
elif save_image_counter > 0:
    print(f"Đã tìm thấy {save_image_counter} ảnh trong thư mục. Bắt đầu từ số {save_image_counter}")

def capture_random_image(silent=False):
    global save_image_counter, last_save_time, notification_label, notification_timer, root
    global missing_image_numbers, current_frame, current_frame_lock, SAVE_HAND_IMAGES
    
    if not silent:
        print("Bắt đầu chụp ảnh ngẫu nhiên...")
    
    # Cảnh báo nếu auto-save chưa bật nhưng vẫn cho phép chụp
    if not SAVE_HAND_IMAGES and not silent:
        print("Auto-save chưa bật, nhưng vẫn cho phép chụp ảnh ngẫu nhiên")
    
    # Kiểm tra khoảng thời gian giữa các lần lưu
    current_time = time.time()
    if current_time - last_save_time < SAVE_INTERVAL:
        print(f"Vui lòng đợi {SAVE_INTERVAL:.1f}s giữa các lần chụp")
        return
    
    # Lấy frame hiện tại (thread-safe)
    frame = None
    with current_frame_lock:
        if current_frame is not None:
            try:
                frame = current_frame.copy()
                print(f"Đã lấy frame: {frame.shape if frame is not None else 'None'}")
            except Exception as e:
                print(f"Lỗi khi copy frame: {e}")
        else:
            print("current_frame là None - chưa có frame nào được lưu")
    
    if frame is None:
        print("Chưa có frame để chụp - vui lòng đợi camera khởi động")
        if notification_label:
            notification_label.config(text=" Chưa có frame!", fg='#ffa500')
            if notification_timer:
                root.after_cancel(notification_timer)
            def restore_status():
                if notification_label:
                    if SAVE_HAND_IMAGES:
                        notification_label.config(text=" Auto-save: ON", fg='#00ff00')
                    else:
                        notification_label.config(text=" Auto-save: OFF", fg='#ff6b6b')
            notification_timer = root.after(2000, restore_status)
        return
    
    try:
        # Tạo thư mục nếu chưa có
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        frame_h, frame_w = frame.shape[:2]
        target_size = 640
        is_resized = False  # Flag để theo dõi xem có resize không
        
        # Xử lý frame: resize hoặc cắt ngẫu nhiên tùy kích thước
        if frame_w < target_size or frame_h < target_size:
            # Frame nhỏ hơn 640x640: resize toàn bộ frame lên 640x640 (giữ tỷ lệ và pad với màu đen)
            print(f"Frame nhỏ ({frame_w}x{frame_h}), tự động resize lên {target_size}x{target_size}")
            is_resized = True
            
            # Tính scale để fit vào 640x640 (giữ tỷ lệ)
            scale = min(target_size / frame_w, target_size / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            
            # Resize frame
            frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Tạo ảnh 640x640 với background đen
            random_crop = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            
            # Đặt ảnh đã resize vào giữa
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            random_crop[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized
            
            random_x = x_offset
            random_y = y_offset
        else:
            # Frame đủ lớn: cắt ngẫu nhiên một vùng 640x640
            # Chọn vị trí ngẫu nhiên để cắt ảnh 640x640
            max_x = frame_w - target_size
            max_y = frame_h - target_size
            
            # Đảm bảo max_x và max_y >= 0
            if max_x < 0:
                max_x = 0
            if max_y < 0:
                max_y = 0
            
            # Random vị trí cắt
            random_x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
            random_y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
            
            # Cắt ảnh 640x640
            random_crop = frame[random_y:random_y+target_size, random_x:random_x+target_size]
        
        if random_crop.size == 0:
            print("Không thể xử lý ảnh")
            return
        
        # Tìm số counter để lưu: ưu tiên dùng số còn thiếu trước
        counter_to_use = None
        
        # Ưu tiên 1: Dùng số từ danh sách các số còn thiếu
        if missing_image_numbers:
            counter_to_use = missing_image_numbers.pop(0)
        else:
            # Ưu tiên 2: Dùng số tiếp theo từ counter
            temp_counter = save_image_counter
            max_attempts = 1000
            attempts = 0
            
            while attempts < max_attempts:
                filename_check = f"{temp_counter}.jpg"
                filepath_check = os.path.join(SAVE_DIR, filename_check)
                
                if not os.path.exists(filepath_check):
                    counter_to_use = temp_counter
                    save_image_counter = temp_counter + 1
                    break
                
                temp_counter += 1
                attempts += 1
            
            if attempts >= max_attempts:
                print(f" Không thể tìm được tên file trống sau {max_attempts} lần thử")
                return
        
        # Tạo tên file và đường dẫn
        filename = f"{counter_to_use}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # Kiểm tra lại một lần nữa để đảm bảo an toàn
        if os.path.exists(filepath):
            print(f"File {filename} đã tồn tại, bỏ qua lần lưu này")
            return
        
        # Lưu ảnh
        cv2.imwrite(filepath, random_crop)
        
        # Cập nhật thời gian
        last_save_time = current_time
        
        # Hiển thị thông báo
        if notification_label:
            notification_label.config(text=f" Đã chụp: {filename}", fg='#00ff00')
        
        # Hủy timer cũ nếu có
        if notification_timer:
            root.after_cancel(notification_timer)
        
        # Tự động ẩn thông báo sau 2 giây
        def restore_auto_save_status():
            if notification_label:
                if SAVE_HAND_IMAGES:
                    notification_label.config(text=" Auto-save: ON", fg='#00ff00')
                else:
                    notification_label.config(text=" Auto-save: OFF", fg='#ff6b6b')
        notification_timer = root.after(2000, restore_auto_save_status)
        
        if is_resized:
            print(f"Đã chụp và resize ảnh: {filepath} (từ {frame_w}x{frame_h} lên {target_size}x{target_size})")
        else:
            print(f"Đã chụp ảnh ngẫu nhiên: {filepath} (vị trí: x={random_x}, y={random_y})")
        
    except Exception as e:
        print(f"Lỗi khi chụp ảnh ngẫu nhiên: {e}")

# Continuous Random Capture
def start_continuous_capture():
    global auto_capture_enabled, auto_capture_job, notification_label, notification_timer, root
    if auto_capture_enabled:
        print("Đã bật chụp liên tục rồi")
        return
    auto_capture_enabled = True
    print(f"Bật chụp ngẫu nhiên liên tục (mỗi {SAVE_INTERVAL}s)")
    
    # Hiển thị thông báo trong UI
    if notification_label:
        notification_label.config(text=f"Chụp liên tục: ON ({SAVE_INTERVAL}s)", fg='#00ff00')
        if notification_timer:
            root.after_cancel(notification_timer)
        def restore_status():
            if notification_label:
                if SAVE_HAND_IMAGES:
                    notification_label.config(text="Auto-save: ON", fg='#00ff00')
                else:
                    notification_label.config(text="Auto-save: OFF", fg='#ff6b6b')
        notification_timer = root.after(3000, restore_status)
    
    _schedule_next_capture()


def stop_continuous_capture():
    global auto_capture_enabled, auto_capture_job, notification_label, notification_timer, root
    auto_capture_enabled = False
    if auto_capture_job and root:
        try:
            root.after_cancel(auto_capture_job)
        except Exception:
            pass
    auto_capture_job = None
    print("Tắt chụp ngẫu nhiên liên tục")
    
    # Hiển thị thông báo trong UI
    if notification_label:
        notification_label.config(text="Chụp liên tục: OFF", fg='#ff6b6b')
        if notification_timer:
            root.after_cancel(notification_timer)
        def restore_status():
            if notification_label:
                if SAVE_HAND_IMAGES:
                    notification_label.config(text="Auto-save: ON", fg='#00ff00')
                else:
                    notification_label.config(text=" Auto-save: OFF", fg='#ff6b6b')
        notification_timer = root.after(2000, restore_status)


def _schedule_next_capture():
    global auto_capture_job
    if not auto_capture_enabled or stop_flag.is_set() or root is None:
        return
    try:
        # Chụp ảnh với silent=True để tránh spam console
        capture_random_image(silent=True)
    except Exception as e:
        print(f"Lỗi khi chụp liên tục: {e}")
    # Lặp lại sau SAVE_INTERVAL (ms)
    delay_ms = max(int(SAVE_INTERVAL * 1000), 10)
    auto_capture_job = root.after(delay_ms, _schedule_next_capture)


def toggle_continuous_capture():
    if auto_capture_enabled:
        stop_continuous_capture()
    else:
        start_continuous_capture()

# Tkinter UI Setup
try:
    root = tk.Tk()
    root.title("MediaPipe Hand Landmarker - Real-time Detection")
    
    # Tính toán kích thước window
    INFO_PANEL_WIDTH = 350
    total_width = WINDOW_WIDTH + INFO_PANEL_WIDTH + 40
    total_height = WINDOW_HEIGHT + 100
    root.geometry(f"{total_width}x{total_height}")
    root.configure(bg='#1e1e1e')  # Dark background
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
    
    # Notification label để hiển thị thông báo lưu ảnh
    notification_label = tk.Label(
        header_frame,
        text="",
        font=('Segoe UI', 11, 'bold'),
        bg='#2d2d2d',
        fg='#00ff00'
    )
    notification_label.pack(side=tk.RIGHT, padx=10, pady=10)
    
    # Hiển thị trạng thái auto-save ban đầu khi khởi động
    if SAVE_HAND_IMAGES:
        notification_label.config(text=" Auto-save: ON", fg='#00ff00')
    else:
        notification_label.config(text=" Auto-save: OFF", fg='#ff6b6b')
    
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
    
    # Video label (sẽ fill toàn bộ container)
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
    
    def toggle_save_images():
        global SAVE_HAND_IMAGES, notification_label, notification_timer
        global missing_image_numbers, save_image_counter
        
        SAVE_HAND_IMAGES = not SAVE_HAND_IMAGES
        
        # Khi bật auto-save, quét lại thư mục để tìm các số còn thiếu
        if SAVE_HAND_IMAGES:
            missing_image_numbers, save_image_counter = scan_missing_image_numbers(SAVE_DIR)
            if missing_image_numbers:
                print(f"Đã tìm thấy {len(missing_image_numbers)} số còn thiếu: {missing_image_numbers}")
                print(f"  Sẽ ưu tiên lấp vào các số này trước khi tiếp tục từ số {save_image_counter}")
            elif save_image_counter > 0:
                print(f"Đã tìm thấy {save_image_counter} ảnh trong thư mục. Bắt đầu từ số {save_image_counter}")
            else:
                print(f"Thư mục trống. Bắt đầu từ số 0")
        
        if notification_label:
            if SAVE_HAND_IMAGES:
                notification_label.config(text=" Auto-save: ON", fg='#00ff00')
            else:
                notification_label.config(text=" Auto-save: OFF", fg='#ff6b6b')
        
        # Hủy timer cũ nếu có
        if notification_timer:
            root.after_cancel(notification_timer)
        
        # Không tự động ẩn trạng thái auto-save (luôn hiển thị để người dùng biết trạng thái)
        # Trạng thái sẽ chỉ bị thay thế tạm thời khi có thông báo lưu ảnh
        
        print(f" Auto-save images: {'ON' if SAVE_HAND_IMAGES else 'OFF'}")
    
    # Bind keyboard shortcuts
    root.bind('<space>', lambda e: toggle_pause())
    root.bind('<KeyPress-x>', lambda e: toggle_save_images())
    root.bind('<KeyPress-X>', lambda e: toggle_save_images())
    
    # Bind phím C để bật/tắt chụp ảnh ngẫu nhiên liên tục
    def handle_capture(e=None):
        print(" Phím C được nhấn - Bật/tắt chụp liên tục")
        try:
            toggle_continuous_capture()
        except NameError:
            print(" Lỗi: Hàm toggle_continuous_capture() chưa được định nghĩa")
        except Exception as ex:
            print(f" Lỗi khi toggle chụp liên tục: {ex}")
    
    # Bind nhiều cách để đảm bảo hoạt động trên mọi hệ thống
    root.bind('<Key-c>', handle_capture)
    root.bind('<Key-C>', handle_capture)
    root.bind('<c>', handle_capture)
    root.bind('<C>', handle_capture)
    root.bind_all('<Key-c>', handle_capture)  # Bind toàn cục
    root.bind_all('<Key-C>', handle_capture)   # Bind toàn cục
    
    # Đảm bảo root luôn nhận focus để nhận keyboard events
    root.focus_set()
    root.focus_force()  # Force focus
    
    # Bind event để đảm bảo focus khi click vào window
    def on_focus_in(e):
        root.focus_set()
    root.bind('<FocusIn>', on_focus_in)
    root.bind('<Button-1>', lambda e: root.focus_set())
    
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
        """Open settings window"""
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
        
        # Separator
        separator1 = tk.Frame(ema_frame, bg='#3d3d3d', height=1)
        separator1.pack(fill=tk.X, pady=15)
        
        # Auto-save Settings
        save_title = tk.Label(
            ema_frame,
            text="Auto-save Settings",
            font=('Segoe UI', 10, 'bold'),
            bg='#252525',
            fg='#ffffff',
            anchor='w'
        )
        save_title.pack(anchor='w', pady=(0, 5))
        
        # SAVE_HAND_IMAGES
        save_images_var = tk.BooleanVar(value=SAVE_HAND_IMAGES)
        tk.Checkbutton(
            ema_frame,
            text="Auto-save Hand Images",
            variable=save_images_var,
            bg='#252525',
            fg='#ffffff',
            selectcolor='#1e1e1e',
            activebackground='#252525',
            activeforeground='#ffffff'
        ).pack(anchor='w', pady=5)
        
        # Save Directory Selection
        save_dir_frame = tk.Frame(ema_frame, bg='#252525')
        save_dir_frame.pack(fill=tk.X, pady=(10, 5))
        
        tk.Label(
            save_dir_frame,
            text="Thư mục lưu ảnh:",
            bg='#252525',
            fg='#aaaaaa',
            anchor='w',
            font=('Segoe UI', 9)
        ).pack(anchor='w', pady=(0, 5))
        
        # Frame chứa đường dẫn và nút
        save_dir_path_frame = tk.Frame(save_dir_frame, bg='#252525')
        save_dir_path_frame.pack(fill=tk.X)
        
        # Label hiển thị đường dẫn (có thể cuộn nếu dài)
        save_dir_label = tk.Label(
            save_dir_path_frame,
            text=SAVE_DIR,
            bg='#1e1e1e',
            fg='#ffffff',
            anchor='w',
            font=('Consolas', 8),
            relief=tk.SUNKEN,
            bd=1,
            padx=5,
            pady=3,
            wraplength=250
        )
        save_dir_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Nút chọn thư mục
        def browse_save_dir():
            global SAVE_DIR, missing_image_numbers, save_image_counter, SAVE_HAND_IMAGES
            selected_dir = filedialog.askdirectory(
                title="Chọn thư mục lưu ảnh",
                initialdir=SAVE_DIR if os.path.exists(SAVE_DIR) else script_dir
            )
            if selected_dir:  # Nếu người dùng chọn thư mục (không cancel)
                SAVE_DIR = selected_dir
                # Cập nhật label hiển thị
                save_dir_label.config(text=SAVE_DIR)
                print(f" Đã chọn thư mục lưu ảnh: {SAVE_DIR}")
                
                # Nếu auto-save đang bật, quét lại thư mục mới để tìm các số còn thiếu
                if SAVE_HAND_IMAGES:
                    missing_image_numbers, save_image_counter = scan_missing_image_numbers(SAVE_DIR)
                    if missing_image_numbers:
                        print(f" Đã tìm thấy {len(missing_image_numbers)} số còn thiếu: {missing_image_numbers}")
                        print(f"  Sẽ ưu tiên lấp vào các số này trước khi tiếp tục từ số {save_image_counter}")
                    elif save_image_counter > 0:
                        print(f" Đã tìm thấy {save_image_counter} ảnh trong thư mục. Bắt đầu từ số {save_image_counter}")
                    else:
                        print(f" Thư mục trống. Bắt đầu từ số 0")
        
        browse_btn = tk.Button(
            save_dir_path_frame,
            text="📁 Chọn",
            command=browse_save_dir,
            bg='#00a8ff',
            fg='#ffffff',
            font=('Segoe UI', 9),
            padx=10,
            pady=3,
            cursor='hand2',
            relief=tk.RAISED,
            bd=1
        )
        browse_btn.pack(side=tk.RIGHT)
        
        # Buttons (luôn ở bottom, không bị che)
        button_frame = tk.Frame(settings_window, bg='#1e1e1e', pady=15, padx=20)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        def apply_settings():
            global NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_PRESENCE_CONFIDENCE, MIN_TRACKING_CONFIDENCE
            global ENABLE_EMA_SMOOTHING, EMA_ALPHA, landmarker, SAVE_HAND_IMAGES, SAVE_DIR
            global missing_image_numbers, save_image_counter
            
            new_num_hands = num_hands_var.get()
            new_min_det = min_det_var.get()
            new_min_track = min_track_var.get()
            new_min_presence = min_presence_var.get()
            new_ema_enable = ema_enable_var.get()
            new_ema_alpha = ema_alpha_var.get()
            new_save_images = save_images_var.get()
            
            # Áp dụng EMA settings ngay
            ENABLE_EMA_SMOOTHING = new_ema_enable
            EMA_ALPHA = new_ema_alpha
            
            # Áp dụng Auto-save settings ngay
            old_save_state = SAVE_HAND_IMAGES
            old_save_dir = SAVE_DIR
            SAVE_HAND_IMAGES = new_save_images
            
            # Nếu thư mục lưu ảnh thay đổi hoặc bật auto-save (từ OFF sang ON), quét lại thư mục
            if (SAVE_HAND_IMAGES and not old_save_state) or (SAVE_HAND_IMAGES and SAVE_DIR != old_save_dir):
                missing_image_numbers, save_image_counter = scan_missing_image_numbers(SAVE_DIR)
                if missing_image_numbers:
                    print(f" Đã tìm thấy {len(missing_image_numbers)} số còn thiếu: {missing_image_numbers}")
                    print(f"  Sẽ ưu tiên lấp vào các số này trước khi tiếp tục từ số {save_image_counter}")
                elif save_image_counter > 0:
                    print(f" Đã tìm thấy {save_image_counter} ảnh trong thư mục. Bắt đầu từ số {save_image_counter}")
                else:
                    print(f" Thư mục trống. Bắt đầu từ số 0")
            
            # Kiểm tra xem có cần recreate landmarker không
            need_recreate = (
                NUM_HANDS != new_num_hands or
                MIN_DETECTION_CONFIDENCE != new_min_det or
                MIN_PRESENCE_CONFIDENCE != new_min_presence or
                MIN_TRACKING_CONFIDENCE != new_min_track
            )
            
            if need_recreate:
                # Update global variables
                NUM_HANDS = new_num_hands
                MIN_DETECTION_CONFIDENCE = new_min_det
                MIN_PRESENCE_CONFIDENCE = new_min_presence
                MIN_TRACKING_CONFIDENCE = new_min_track
                
                # Recreate landmarker với options mới
                try:
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
                    
                    print(f" Landmarker recreated with new settings:")
                    print(f"  NUM_HANDS={NUM_HANDS}, MIN_DET={MIN_DETECTION_CONFIDENCE:.2f}, "
                          f"MIN_PRESENCE={MIN_PRESENCE_CONFIDENCE:.2f}, MIN_TRACK={MIN_TRACKING_CONFIDENCE:.2f}")
                except Exception as e:
                    print(f" Error recreating landmarker: {e}")
                    return
            
            print(f" Settings applied:")
            print(f"  NUM_HANDS={NUM_HANDS}, MIN_DET={MIN_DETECTION_CONFIDENCE:.2f}, "
                  f"MIN_PRESENCE={MIN_PRESENCE_CONFIDENCE:.2f}, MIN_TRACK={MIN_TRACKING_CONFIDENCE:.2f}")
            print(f"  EMA={ENABLE_EMA_SMOOTHING}, ALPHA={EMA_ALPHA:.2f}")
            print(f"  Auto-save Images={SAVE_HAND_IMAGES}")
            print(f"  Save Directory={SAVE_DIR}")
        
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
    
    # Capture button in header (để bật/tắt chụp liên tục)
    capture_btn = tk.Button(
        header_frame,
        text="📷 Chụp liên tục (C)",
        command=toggle_continuous_capture,
        bg='#2d2d2d',
        fg='#00ff00',
        font=('Segoe UI', 9),
        relief=tk.FLAT,
        padx=10,
        pady=5,
        cursor='hand2',
        activebackground='#3d3d3d',
        activeforeground='#00ff00'
    )
    capture_btn.pack(side=tk.RIGHT, padx=5)
    
    print(" Tkinter UI initialized")
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

def save_hand_image(frame, min_x, min_y, max_x, max_y):
    global save_image_counter, last_save_time, notification_label, notification_timer, root
    global missing_image_numbers
    
    # Kiểm tra khoảng thời gian giữa các lần lưu
    current_time = time.time()
    if current_time - last_save_time < SAVE_INTERVAL:
        return
    
    try:
        # Tạo thư mục nếu chưa có
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # Cắt ảnh từ bounding box (thêm padding nhỏ để không bị cắt)
        padding = 20
        frame_h, frame_w = frame.shape[:2]
        
        # Tính toán tọa độ với padding
        crop_x1 = max(0, min_x - padding)
        crop_y1 = max(0, min_y - padding)
        crop_x2 = min(frame_w, max_x + padding)
        crop_y2 = min(frame_h, max_y + padding)
        
        # Cắt ảnh
        hand_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if hand_crop.size == 0:
            return
        
        # Resize về 640x640 (giữ tỷ lệ và pad với màu đen nếu cần)
        target_size = 640
        h, w = hand_crop.shape[:2]
        
        # Tính scale để fit vào 640x640
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        hand_resized = cv2.resize(hand_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Tạo ảnh 640x640 với background đen
        hand_final = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Đặt ảnh đã resize vào giữa
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        hand_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = hand_resized
        
        # Tìm số counter để lưu: ưu tiên dùng số còn thiếu trước
        counter_to_use = None
        
        # Ưu tiên 1: Dùng số từ danh sách các số còn thiếu
        if missing_image_numbers:
            counter_to_use = missing_image_numbers.pop(0)  # Lấy số đầu tiên và xóa khỏi list
        else:
            # Ưu tiên 2: Dùng số tiếp theo từ counter
            # Kiểm tra xem số này đã tồn tại chưa (phòng trường hợp có file mới được thêm vào từ bên ngoài)
            temp_counter = save_image_counter
            max_attempts = 1000
            attempts = 0
            
            while attempts < max_attempts:
                filename_check = f"{temp_counter}.jpg"
                filepath_check = os.path.join(SAVE_DIR, filename_check)
                
                # Nếu file chưa tồn tại, dùng số này
                if not os.path.exists(filepath_check):
                    counter_to_use = temp_counter
                    save_image_counter = temp_counter + 1  # Cập nhật counter cho lần sau
                    break
                
                # Nếu file đã tồn tại, thử số tiếp theo
                temp_counter += 1
                attempts += 1
            
            if attempts >= max_attempts:
                print(f" Không thể tìm được tên file trống sau {max_attempts} lần thử")
                return
        
        # Tạo tên file và đường dẫn
        filename = f"{counter_to_use}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # Kiểm tra lại một lần nữa để đảm bảo an toàn (phòng trường hợp có race condition)
        if os.path.exists(filepath):
            print(f" File {filename} đã tồn tại, bỏ qua lần lưu này")
            return
        
        # Lưu ảnh
        cv2.imwrite(filepath, hand_final)
        
        # Cập nhật thời gian
        last_save_time = current_time
        
        # Hiển thị thông báo
        if notification_label:
            notification_label.config(text=f" Đã lưu: {filename}", fg='#00ff00')
        
        # Hủy timer cũ nếu có
        if notification_timer:
            root.after_cancel(notification_timer)
        
        # Tự động ẩn thông báo sau 2 giây và quay lại hiển thị trạng thái auto-save
        def restore_auto_save_status():
            if notification_label:
                if SAVE_HAND_IMAGES:
                    notification_label.config(text=" Auto-save: ON", fg='#00ff00')
                else:
                    notification_label.config(text=" Auto-save: OFF", fg='#ff6b6b')
        notification_timer = root.after(2000, restore_auto_save_status)
        
        print(f" Đã lưu ảnh: {filepath}")
        
    except Exception as e:
        print(f" Lỗi khi lưu ảnh: {e}")

def draw_keypoints(frame, keypoints, color=(0, 255, 255), radius=3, conf_threshold=0.3):
    """
    Vẽ keypoints lên frame (tối ưu cho real-time với OpenCV direct calls)
    
    Performance: Custom OpenCV nhanh hơn MediaPipe official draw_landmarks vì:
    - Không có protobuf conversion overhead
    - Direct C++ OpenCV backend
    - Có thể tối ưu validation và bounds checking
    
    Args:
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
    
    Args:
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
    global cached_container_size, cached_metrics_values, is_paused, notification_timer
    global current_frame, current_frame_lock
    
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
        
        # Lưu frame hiện tại để có thể chụp ảnh ngẫu nhiên (thread-safe)
        with current_frame_lock:
            current_frame = frame_original.copy()
        
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
            if inference_time > 0:
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
        current_inference_fps = 1.0 / inference_time if inference_time > 0 else None
        avg_fps_display = moving_avg(fps_list)
        avg_inference_fps_display = moving_avg(inference_fps_list)
        avg_input_fps_display = moving_avg(input_fps_list)
        avg_display_latency = moving_avg(display_latencies)
        avg_inference_time = moving_avg(inference_times)
        current_input_fps = input_fps_list[-1] if input_fps_list else None
        
        # Visualization (MediaPipe hand landmarks + bounding box)
        annotated_frame = frame_original.copy()
        
        if result and result.hand_landmarks:
            try:
                # Cleanup old EMA state (prevent memory leak)
                current_hand_indices = set(range(len(result.hand_landmarks)))
                cleanup_old_ema_state(current_hand_indices)
                
                for hand_idx, landmarks in enumerate(result.hand_landmarks):
                    # landmarks: list 21 điểm, mỗi điểm có x, y (normalized)
                    # Validate và clamp x, y trong khoảng [0, 1] để tránh crash nếu MediaPipe trả về giá trị lỗi
                    landmarks_array = np.array([
                        [max(0.0, min(1.0, lm.x)) * frame_w, max(0.0, min(1.0, lm.y)) * frame_h, 1.0] 
                        for lm in landmarks
                    ], dtype=np.float32)
                    
                    # Apply EMA smoothing to reduce jitter
                    landmarks_array = apply_ema_smoothing(hand_idx, landmarks_array, alpha=EMA_ALPHA)
                    
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
                        handedness_score = float(score)
                    # Nếu độ tin cậy handedness quá thấp thì bỏ qua (không vẽ tay)
                    # Loại bỏ các detection không chắc chắn (có thể là false positive)
                    if handedness_score < HANDEDNESS_SCORE_THRESHOLD:
                        continue

                    color = get_track_color(hand_idx)  # dùng index tay làm ID tạm

                    label = f"ID:{hand_idx} {handedness_label}"
                    
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
                    
                    # Tự động lưu ảnh bàn tay (raw data - không có keypoints)
                    if SAVE_HAND_IMAGES:
                        save_hand_image(frame_original, min_x, min_y, max_x, max_y)

            except Exception as e:
                print(f" Error drawing MediaPipe results: {e}")
        
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
                        # Tạo mới nếu size thay đổi
                        video_label.photo_image = ImageTk.PhotoImage(image=pil_image)
                        video_label.photo_image_size = pil_image.size
                        video_label.config(image=video_label.photo_image, text="")
                except Exception:
                    # Fallback: tạo mới PhotoImage nếu có lỗi
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