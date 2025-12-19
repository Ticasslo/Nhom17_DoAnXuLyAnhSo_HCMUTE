# Nhóm 17 – Đồ án Xử Lý Ảnh Số – HCMUTE

Thắng, Đăng, Nhân, Kha xin chào.

# Models
https://drive.google.com/drive/folders/13sTNsJhthwzoIVg4f3iuINucdpt93TFq?usp=sharing
# Kích hoạt thư viện venv:
& D:\Nhom17_DoAnXuLyAnhSo_HCMUTE\.venv\Scripts\Activate.ps1

## 1. Mục tiêu hệ thống

Hệ thống quạt thông minh điều khiển bằng cử chỉ tay, gồm 3 phần chính:

- **ESP32-CAM**: quay video và stream qua WiFi (`http://esp32-cam.local/stream`).
- **UI Python (Tkinter + MediaPipe + TF)**: nhận video, nhận diện tay, phân loại cử chỉ, gửi lệnh điều khiển qua MQTT.
- **ESP32 Receiver**: nhận lệnh từ MQTT, điều khiển **LED, quạt DC (tốc độ) và 2 servo (hướng gió)**.

Toàn bộ pipeline:  
**Camera → Python nhận diện cử chỉ → MQTT → ESP32 Receiver → Quạt + LED + Servo**.

---

## 2. Cấu trúc thư mục

```text
Nhom17_DoAnXuLyAnhSo_HCMUTE/
├─ esp32cam_sender/              # Code ESP32-CAM (stream MJPEG + mDNS)
│   └─ esp32cam_sender.ino
├─ esp32_receiver/               # Code ESP32 nhận lệnh MQTT + điều khiển phần cứng
│   └─ esp32_receiver.ino
├─ models/
│   ├─ hand_landmarker.task      # Model MediaPipe (hand landmarks) dùng chung
│   └─ SavedModel/               # Model phân loại cử chỉ (TensorFlow SavedModel)
│       ├─ saved_model_best/     # Model TF SavedModel dùng cho realtime
│       └─ metadata.pkl          # Thông tin labels, num_features, ...
├─ UI/
│   ├─ Step0_Requirement/        # Kiểm tra version thư viện (import_check_library.py, inspect_model_metadata.py)
│   ├─ Step1_DataCollection/     # Tool thu thập ảnh bàn tay (MediaPipe + Tkinter)
│   ├─ Step2_DataExtraction_Normalization_Augmentation/
│   │   └─ data_extraction_and_augmentation.py  # Trích landmarks + ISP + augment
│   ├─ Step5_RealtimePredictiion/
│   │   ├─ mqtt_publisher.py     # Module MQTT (Edge-Trigger + Hold-to-Repeat + mDNS)
│   │   └─ tkinter_detection_classification_esp32.py  # UI realtime chính
│   └─ TemplateUI/               # Các template Tkinter tham khảo
│       ├─ tkinter_template_detection.py
│       └─ tkinter_template_detection_classification.py
└─ README.md                     # (file hiện tại)
```

---

## 3. Kiến trúc & luồng dữ liệu

### 3.1 ESP32-CAM (`esp32cam_sender/esp32cam_sender.ino`)

- Kết nối WiFi STA với SSID `ilovehcmute`.
- Khởi tạo camera OV2640 với cấu hình:
  - `FRAME_SIZE = HVGA (480x320)` – cân bằng chất lượng & FPS.
  - `JPEG_QUALITY = 35`, `FB_COUNT = 3` – tối ưu RAM & tốc độ.
- Chạy HTTP server với 1 endpoint:
  - `GET /stream`: trả về **MJPEG multipart** (chuỗi các frame JPEG).
  - Giới hạn ~15 FPS để tránh nóng/treo khi chạy lâu.
- Bật **mDNS** với hostname `esp32-cam`:
  - Người dùng có thể truy cập `http://esp32-cam.local/stream` từ Python UI / trình duyệt.
- Có FreeRTOS task `wifiTask` quản lý reconnect WiFi (ổn định khi chạy nhiều giờ).

### 3.2 Python UI Realtime (`UI/Step5_RealtimePredictiion/`)

#### a. `mqtt_publisher.py`

- Tự động lấy **IP LAN** của máy tính (qua socket) → làm `MQTT_BROKER_HOST`.
- Quảng bá dịch vụ MQTT qua **mDNS** (`_mqtt._tcp.local.`) để ESP32 có thể tìm được broker mà **không cần IP cố định**.
- Cơ chế gửi lệnh:
  - **Edge-trigger**: chỉ gửi khi cử chỉ **thay đổi**, tránh spam lệnh giống nhau.
  - **Hold-to-Repeat**: cho 4 lệnh điều hướng `FanLeft`, `FanRight`, `FanUp`, `FanDown`:
    - Nếu giữ tay lâu, sau mỗi `1.5s` sẽ gửi lại 1 lệnh để quạt nhích thêm 1 nấc.
- Log chi tiết: broker, topic, payload, mã lỗi → dễ debug.

#### b. `tkinter_detection_classification_esp32.py`

- Nhận video từ `SOURCE = "http://esp32-cam.local/stream"`.
- Dùng **MediaPipe Hand Landmarker** để lấy 21 keypoints mỗi tay.
- Chuẩn hóa landmarks (relative + scale + orientation `Y_hand`, `X_hand`) giống **y hệt** bên script trích data.
- Dùng model TensorFlow SavedModel để **phân loại cử chỉ** (Fan/Light/Start/...).
- Cơ chế chống nhiễu:
  - Ngưỡng `GESTURE_REJECT_THRESHOLD`, **entropy**, **orientation magnitude** filter.
  - Lớp `GestureVoter` yêu cầu:
    - Min votes
    - Thời gian giữ tay tối thiểu
    - Tỷ lệ phiếu đồng ý ≥ 80%
- **Cơ chế “Start” kích hoạt hệ thống**:
  - Ban đầu: `SYSTEM: IDLE (Wait for 'Start')` → **mọi lệnh đều bị bỏ qua** trừ `Start`.
  - Khi nhận lệnh `Start`:
    - Chuyển sang `ACTIVE`, log: `SYSTEM ACTIVATED by Start gesture`.
    - Bắt buộc gửi MQTT `Start` (với `force=True`) để ESP32 beep xác nhận.
  - Nếu **không có gesture mới** trong `20s`:
    - Tự về `IDLE`, log: `SYSTEM DEACTIVATED due to 20.0s inactivity`.
- Khi gửi MQTT:
  - Tự động **bỏ prefix** `A_RH_`, `A_LH_`, `S_` → ESP32 nhận lệnh “sạch” như `Light1On`, `FanLeft`, `FanSpeed2`…

### 3.3 Data Extraction & ISP (`UI/Step2_DataExtraction_Normalization_Augmentation/`)

- Script `data_extraction_and_augmentation.py`:
  - Đọc ảnh từ `dataset/` theo từng folder gesture.
  - Áp dụng **ISP** (`tien_xu_ly`):
    - Resize 640x640.
    - White balance nhẹ (giới hạn scale 0.9–1.1) → giữ màu da tự nhiên.
    - Adaptive gamma → cân bằng sáng/tối.
    - Gaussian blur 3x3 → giảm nhiễu để landmarks ổn định.
  - Dùng MediaPipe để lấy landmarks, normalize giống realtime.
  - Augmentation:
    - Chỉ áp dụng cho **symmetric gestures** (nhãn dạng `S_...`) bằng cách **flip ảnh**.
  - Validate **handedness** (A*LH*_ phải là Left, A*RH*_ phải là Right) → bỏ mọi sample sai tay.
  - Lưu kết quả vào `dataset.csv` với đầy đủ features + label + cờ handedness.

### 3.4 ESP32 Receiver (`esp32_receiver/esp32_receiver.ino`)

- Tự connect WiFi + dùng **mDNS** (`MDNS.queryService("mqtt","tcp")`) để tìm broker.
- Subcribe topic `gesture/command`.
- Parse JSON `{ "gesture": "...", "confidence": ..., ... }` và điều khiển:
  - **LED 1/2**: `Light1On/Off`, `Light2On/Off`.
  - **Quạt DC** (qua L298N):
    - `FanOff`, `FanSpeed1` (=30%), `FanSpeed2` (=65%), `FanSpeed3` (=100%).
    - Có ramp mượt để tránh giật máy.
  - **Servo1 (trái–phải)**: góc `{50, 90, 130}` tương ứng với FanLeft/FanRight (3 nấc).
  - **Servo2 (lên–xuống)**: góc `{0, 35, 70}` tương ứng FanUp/FanDown (3 nấc).
- **Buzzer**:
  - `Start`: beep “khởi động” đặc biệt.
  - Mọi lệnh điều khiển khác: beep ngắn **trước khi** thực hiện lệnh.
- Log Serial chi tiết để debug.

### 3.5 Cài đặt Mosquitto MQTT Broker

Hệ thống cần **Mosquitto MQTT Broker** để kết nối giữa Python UI và ESP32 Receiver.

#### Windows

1. **Tải Mosquitto**:

   - Truy cập: https://mosquitto.org/download/
   - Tải file `.msi` cho Windows (ví dụ: `mosquitto-2.x.x-install-windows-x64.exe`)

2. **Cài đặt**:

   - Chạy file `.msi`, chọn "Complete installation"
   - Cài đặt vào thư mục mặc định (thường là `C:\Program Files\mosquitto\`)

3. **Cấu hình** (tùy chọn):

   - Mở file `C:\Program Files\mosquitto\mosquitto.conf` bằng Notepad (cần quyền Admin)
   - Đảm bảo có các dòng sau để cho phép kết nối từ mạng LAN:
     ```
     listener 1883
     allow_anonymous true
     ```
   - **Lưu ý**: `allow_anonymous true` chỉ dùng cho môi trường phát triển/test. Với production nên dùng username/password.

4. **Chạy Mosquitto**:

   - Mở **Command Prompt** hoặc **PowerShell** với quyền Admin
   - Chạy lệnh:
     ```bash
     "C:\Program Files\mosquitto\mosquitto.exe" -c "C:\Program Files\mosquitto\mosquitto.conf"
     ```
   - Hoặc nếu đã thêm vào PATH:
     ```bash
     mosquitto -c mosquitto.conf
     ```
   - Nếu thành công, bạn sẽ thấy log: `mosquitto version x.x.x starting`

5. **Kiểm tra**:

   - Mở terminal khác, chạy:
     ```bash
     mosquitto_sub -h localhost -t "test" -v
     ```
   - Mở terminal thứ 3, chạy:
     ```bash
     mosquitto_pub -h localhost -t "test" -m "Hello MQTT"
     ```
   - Nếu thấy "test Hello MQTT" ở terminal đầu tiên → Mosquitto hoạt động đúng.

6. **Chạy tự động khi khởi động** (tùy chọn):
   - Mở **Services** (Win+R → `services.msc`)
   - Tìm service "Mosquitto Broker"
   - Click chuột phải → **Properties** → **Startup type**: chọn **Automatic**
   - Click **Start** để chạy ngay

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients

# Khởi động service
sudo systemctl start mosquitto
sudo systemctl enable mosquitto  # Tự động chạy khi khởi động

# Kiểm tra
mosquitto_sub -h localhost -t "test" -v
```

#### macOS

```bash
# Dùng Homebrew
brew install mosquitto

# Khởi động service
brew services start mosquitto

# Kiểm tra
mosquitto_sub -h localhost -t "test" -v
```

**Lưu ý quan trọng**:

- Mosquitto phải **chạy trên máy tính** (không phải ESP32)
- Python UI sẽ tự động lấy IP LAN của máy tính làm broker host
- ESP32 Receiver sẽ tự tìm broker qua mDNS (không cần cấu hình IP tĩnh)

---

## 4. Cách chạy nhanh

1. **Nạp ESP32-CAM**

   - Mở `esp32cam_sender/esp32cam_sender.ino` trong Arduino IDE.
   - Chọn đúng board ESP32-S3 + config PSRAM.
   - Nạp code, mở Serial Monitor để xem IP và log mDNS (`esp32-cam.local`).

2. **Nạp ESP32 Receiver**

   - Mở `esp32_receiver/esp32_receiver.ino`.
   - Chỉnh lại SSID/password nếu cần.
   - Nạp code, mở Serial Monitor kiểm tra:
     - WiFi connected
     - `mDNS: Đã tìm thấy Broker` (sau khi broker chạy)

3. **Chạy Mosquitto Broker trên PC**

   - Cấu hình để listen trên IP LAN (ví dụ `192.168.x.x`) và port `1883`.
   - Kiểm tra log: phải thấy ESP32 receiver connect & subscribe `gesture/command`.

4. **Chạy UI realtime**

   - Mở terminal tại `UI/Step5_RealtimePredictiion/`.
   - (Khuyến nghị dùng Python 3.10–3.11, cài đủ package như trong `Step0_Requirement/import_check_library.py`)
   - Chạy: `python tkinter_detection_classification_esp32.py`
   - Kiểm tra console:
     - Model + metadata load OK.
     - MQTT: kết nối thành công tới broker.
     - Dòng `Starting MediaPipe Hand Landmarker with source: http://esp32-cam.local/stream`.

5. **Sử dụng**
   - Đứng trước camera, đưa tay:
     - Hiển thị status `SYSTEM: IDLE (Wait for 'Start')`.
     - Làm gesture `Start` → chuyển sang `SYSTEM: ACTIVE`, ESP32 beep.
   - Thực hiện các gesture `Light1On`, `FanSpeed1/2/3`, `FanLeft/Right`, `FanUp/Down`, ...
   - Nếu không cử chỉ nào được nhận diện thêm trong **20 giây**, hệ thống tự về IDLE.

---

## 5. Ghi chú kỹ thuật

- **Độ ổn định**:
  - Camera: giới hạn FPS và tắt power-save WiFi để chạy lâu không treo.
  - ESP32 Receiver: dùng mDNS nên không phụ thuộc IP tĩnh của PC.
  - UI: nhiều lớp filter + voting để tránh lệnh sai do nhiễu.
- **Đào tạo model**:
  - Dataset được tiền xử lý (ISP) + augmentation cho symmetric gestures.
  - Đảm bảo **normalize features giống hệt** giữa training và realtime.
