import os
import time
import cv2
import warnings
import threading
from queue import Queue, Empty, Full
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

import mediapipe as mp
from mediapipe.tasks.python import vision

# Giảm warning log
warnings.filterwarnings("ignore", category=UserWarning)

# ========== 1. CONFIGURATION ==========
# Camera settings (ESP32-S3 OV2640 stream)
# Đặt IP của ESP32-CAM (web stream ở port 81 theo sketch esp32cam.ino)
ESP32_STREAM_URL = os.getenv("ESP32_STREAM_URL", "http://192.168.24.179:80/stream")
SOURCE = ESP32_STREAM_URL

# Performance settings
NUM_HANDS = 4  # 2 người (mỗi người 2 tay) - có thể giảm xuống 2 nếu chỉ cần 1 người
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

# Stream reconnect settings
STREAM_RETRY_INTERVAL = 2.0  # Thời gian chờ giữa các lần retry (giây)
STREAM_MAX_RETRIES = 10  # Số lần retry tối đa (0 = retry vô hạn)
# =======================================

# ---------- 2. MediaPipe Hand Landmarker ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
# TODO: tải model official từ docs và đặt cạnh script này
# Ví dụ: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
HAND_LANDMARKER_MODEL_PATH = os.path.join(script_dir, "hand_landmarker.task")

if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy model MediaPipe: {HAND_LANDMARKER_MODEL_PATH}\n"
        f"Vui lòng tải file .task (hand_landmarker.task) về và đặt cạnh script này."
    )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Tối ưu hiệu suất cho Windows:
# - Lưu ý: MediaPipe Python trên Windows KHÔNG hỗ trợ GPU delegate
# - Các tối ưu đã áp dụng:
#   1. Warm-up model (giảm latency spike)
#   2. Tối ưu số lượng hands detect (giảm num_hands nếu không cần nhiều)
#   3. Multi-threading
#   4. Tối ưu confidence thresholds
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

# ---------- EMA Smoothing State ----------
# EMA (Exponential Moving Average) state for each hand
# Structure: {hand_idx: {'landmarks': array, 'last_seen': timestamp}}
ema_state = {}

def apply_ema_smoothing(hand_idx, current_landmarks, alpha=EMA_ALPHA):
    """
    Apply Exponential Moving Average smoothing to landmarks
    
    EMA formula: smoothed_t = alpha * current + (1 - alpha) * smoothed_t-1
    
    Benefits:
    - Memory efficient: Only stores 1 previous value (vs N frames for moving average)
    - Computation efficient: Only 1 multiplication + 1 addition per keypoint
    - Adaptive: Automatically adjusts to motion speed
    - Lower latency: ~16-20ms lag vs ~33-50ms for moving average
    
    Args:
        hand_idx: Hand index (for tracking across frames)
        current_landmarks: Current frame landmarks (21, 3) numpy array
        alpha: Smoothing factor (0.0=max smooth, 1.0=no smooth)
               Recommended: 0.1 (very smooth), 0.3 (balanced), 0.5 (responsive)
    
    Returns:
        smoothed_landmarks: EMA-smoothed landmarks (21, 3) numpy array
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
    """
    Remove EMA state for hands that haven't been seen recently
    Call this periodically to avoid memory leak
    
    Args:
        current_hand_indices: Set of hand indices detected in current frame
        max_age_seconds: Remove hands not seen for this many seconds
    """
    global ema_state
    current_time = time.time()
    
    # Remove hands not in current frame AND not seen for >max_age_seconds
    ema_state = {
        idx: state for idx, state in ema_state.items() 
        if idx in current_hand_indices or (current_time - state['last_seen']) < max_age_seconds
    }

# ---------- 3. Queue & threading setup ----------
stream_url = SOURCE
target_fps = 30.0

print("=" * 60)
print("CAMERA MODE - MediaPipe Hand Landmarker (keypoints + bbox)")

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
    """
    Thread 1: Đọc frame từ camera và đưa vào queue.
    
    Tối ưu: Dùng MSMF backend trên Windows (nhanh hơn DirectShow).
    Fallback về default nếu không support.
    Có retry logic để tự động reconnect khi stream bị mất.
    """
    global queue_drop_count
    
    def open_stream():
        """Mở stream và config OpenCV settings"""
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer để giảm latency
            cap.set(cv2.CAP_PROP_FPS, target_fps)  # Set FPS nếu camera support
        return cap
    
    # Thử mở stream lần đầu
    cap = open_stream()
    if not cap.isOpened():
        print("✗ Error: Cannot open video source")
        print(f"  → Kiểm tra ESP32 IP: {stream_url}")
        print("  → Đảm bảo ESP32 đã khởi động và stream đang chạy")
        stop_flag.set()
        return
    
    frame_id = 0
    retry_count = 0
    consecutive_failures = 0
    
    while not stop_flag.is_set():
        ret, frame = cap.read()
        
        # Kiểm tra cả ret và frame (sau reconnect có thể ret=True nhưng frame=None)
        if not ret or frame is None:
            consecutive_failures += 1
            
            # Nếu fail liên tiếp nhiều lần, thử reconnect
            if consecutive_failures >= 3:
                print(f"⚠ Stream lost. Attempting to reconnect... (attempt {retry_count + 1})")
                
                # Đóng stream cũ
                try:
                    cap.release()
                except Exception:
                    pass
                
                # Kiểm tra max retries
                if STREAM_MAX_RETRIES > 0 and retry_count >= STREAM_MAX_RETRIES:
                    print(f"✗ Max retries ({STREAM_MAX_RETRIES}) reached. Stopping stream.")
                    break
                
                # Đợi trước khi retry
                time.sleep(STREAM_RETRY_INTERVAL)
                
                # Thử mở lại stream
                cap = open_stream()
                if cap.isOpened():
                    print("✓ Stream reconnected successfully!")
                    retry_count = 0
                    consecutive_failures = 0
                    # Đợi một chút để stream sẵn sàng trước khi đọc frame
                    time.sleep(0.5)
                    continue  # Tiếp tục vòng lặp để đọc frame mới
                else:
                    retry_count += 1
                    consecutive_failures = 0  # Reset để đếm lại
                    continue
            else:
                # Fail ít lần, đợi ngắn rồi thử lại
                time.sleep(0.1)
                continue
        
        # Đọc frame thành công - stream đã hoạt động lại
        consecutive_failures = 0
        # Reset retry_count vì stream đã hoạt động lại (có thể tự recover hoặc reconnect thành công)
        if retry_count > 0:
            retry_count = 0
        
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
    
    # Cleanup
    try:
        cap.release()
    except Exception:
        pass
    stop_flag.set()
    print("Thread 1 (Frame Grabber) stopped")


def hand_landmarker_thread():
    """
    Thread 2: Lấy frame từ queue, chạy MediaPipe Hand Landmarker (VIDEO mode)
    và đẩy kết quả (keypoints + handedness) sang detection_queue.
    
    MediaPipe yêu cầu RGB format và Image wrapper.
    Tối ưu: Set flags.writeable = False để tăng tốc (MediaPipe không modify image).
    """
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
                print(f"✗ Error in HandLandmarker thread processing: {e}")
            finally:
                # Đảm bảo task_done() chỉ được gọi 1 lần cho mỗi frame
                frame_queue.task_done()
            
        except Empty:
            if stop_flag.is_set():
                break
            continue
        except Exception as e:
            print(f"✗ Error in HandLandmarker thread (queue get): {e}")
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

# ---------- 4. Hiển thị real-time ----------
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

# Cache container size để tránh gọi winfo_width/height mỗi frame (performance)
cached_container_size = {'w': WINDOW_WIDTH, 'h': WINDOW_HEIGHT, 'last_scale': 1.0, 'last_w': 0, 'last_h': 0}
cached_metrics_values = {}  # Cache metrics values để chỉ update khi thay đổi

# UI State
is_paused = False

# ---------- Tkinter UI Setup ----------
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
    
    # ========== HEADER ==========
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
    
    # ========== MAIN CONTENT AREA ==========
    main_frame = tk.Frame(root, bg='#1e1e1e')
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # ========== LEFT SIDE: INFO PANEL ==========
    info_panel = tk.Frame(main_frame, bg='#252525', width=INFO_PANEL_WIDTH)
    info_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
    info_panel.pack_propagate(False)
    
    # Title cho info panel
    info_title = tk.Label(
        info_panel,
        text="Performance Metrics",
        font=('Segoe UI', 12, 'bold'),
        bg='#252525',
        fg='#ffffff',
        anchor='w'
    )
    info_title.pack(fill=tk.X, padx=15, pady=(15, 10))
    
    # Metrics container với vertical layout
    metrics_container = tk.Frame(info_panel, bg='#252525')
    metrics_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
    
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
    
    # Tạo vertical layout cho metrics
    for key, label, color in metric_configs:
        # Metric container
        metric_frame = tk.Frame(metrics_container, bg='#252525')
        metric_frame.pack(fill=tk.X, pady=8)
        
        # Label name
        name_label = tk.Label(
            metric_frame,
            text=f"{label}:",
            font=('Segoe UI', 9),
            bg='#252525',
            fg='#aaaaaa',
            anchor='w'
        )
        name_label.pack(anchor='w', padx=(0, 5))
        
        # Value label
        value_label = tk.Label(
            metric_frame,
            text="--",
            font=('Consolas', 11, 'bold'),
            bg='#252525',
            fg=color,
            anchor='w'
        )
        value_label.pack(anchor='w')
        
        metrics_labels[key] = value_label
    
    # ========== RIGHT SIDE: VIDEO DISPLAY ==========
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
    
    # ========== KEYBOARD SHORTCUTS ==========
    def toggle_pause():
        """Toggle pause/resume detection"""
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
        print("\nStopped by user (closed window)")
        stop_flag.set()
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # ========== SETTINGS PANEL ==========
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
        
        # ========== COLUMN 1: Performance Settings ==========
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
        
        # ========== COLUMN 2: EMA Settings ==========
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
            global ENABLE_EMA_SMOOTHING, EMA_ALPHA, landmarker
            
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
                    
                    print(f"✓ Landmarker recreated with new settings:")
                    print(f"  NUM_HANDS={NUM_HANDS}, MIN_DET={MIN_DETECTION_CONFIDENCE:.2f}, "
                          f"MIN_PRESENCE={MIN_PRESENCE_CONFIDENCE:.2f}, MIN_TRACK={MIN_TRACKING_CONFIDENCE:.2f}")
                except Exception as e:
                    print(f"✗ Error recreating landmarker: {e}")
                    return
            
            print(f"✓ Settings applied:")
            print(f"  NUM_HANDS={NUM_HANDS}, MIN_DET={MIN_DETECTION_CONFIDENCE:.2f}, "
                  f"MIN_PRESENCE={MIN_PRESENCE_CONFIDENCE:.2f}, MIN_TRACK={MIN_TRACKING_CONFIDENCE:.2f}")
            print(f"  EMA={ENABLE_EMA_SMOOTHING}, ALPHA={EMA_ALPHA:.2f}")
        
        def close_settings():
            """Close settings window"""
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
    
    print("✓ Tkinter UI initialized")
except Exception as e:
    raise RuntimeError(f"Không thể khởi tạo Tkinter UI: {e}") from e

current_photo = None

# ---------- Helper Functions ----------
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

# ---------- Main Update Loop ----------
def update_frame():
    """Update frame trong Tkinter UI (chạy trong mainloop)"""
    global frame_count, total_objects, prev_display_time, prev_capture_time
    global latest_detection, current_photo
    global inference_fps_list, inference_times, input_fps_list, fps_list, frame_intervals, display_latencies
    global cached_container_size, cached_metrics_values, is_paused
    
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
            print(f"⚠ Error getting frame dimensions: {e}")
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

            except Exception as e:
                print(f"⚠ Error drawing MediaPipe results: {e}")
        
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
            print(f"⚠ Error updating Tkinter UI: {e}")
        
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
        print(f"✗ Error in update_frame: {e}")
        if not stop_flag.is_set():
            root.after(10, update_frame)

# Chạy Tkinter main loop
root.after(10, update_frame)
root.mainloop()

# ---------- 5. Cleanup & Summary ----------
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