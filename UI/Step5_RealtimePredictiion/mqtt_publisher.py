import paho.mqtt.client as mqtt
import json
import time
import threading
import socket
from typing import Optional
from zeroconf import ServiceInfo, Zeroconf

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG MQTT
# ==========================================
# Sử dụng localhost vì Python chạy cùng máy với Mosquitto Broker
MQTT_BROKER_HOST = "localhost" 
# Cổng mặc định của giao thức MQTT
MQTT_BROKER_PORT = 1883
# Thời gian giữ kết nối (giây)
MQTT_KEEPALIVE = 60
# Tên "đường ống" để gửi lệnh cử chỉ
MQTT_TOPIC_GESTURE = "gesture/command"
# Tên định danh của Python trên hệ thống MQTT
MQTT_CLIENT_ID = "gesture_detector_publisher"
# Chất lượng dịch vụ: 1 (đảm bảo lệnh đến nơi ít nhất 1 lần)
MQTT_QOS = 1

# Khoảng thời gian tối thiểu giữa 2 lần gửi cùng 1 lệnh (giây)
# (khi ép buộc gửi lại lệnh trùng bằng tham số force=True)
GESTURE_COOLDOWN_SECONDS = 1.0 

# DANH SÁCH LỆNH CHO PHÉP LẶP LẠI (Hold-to-Repeat)
# Các lệnh này sẽ tự động gửi lại sau một khoảng thời gian nếu bạn vẫn giữ tay
REPEATABLE_GESTURES = ["FanLeft", "FanRight", "FanUp", "FanDown"]
REPEAT_COOLDOWN_SECONDS = 1.5  # Giữ tay 1.5 giây sẽ nhích thêm 1 nấc quay

class MQTTPublisher:
    def __init__(self, 
                 broker_host: str = MQTT_BROKER_HOST,
                 broker_port: int = MQTT_BROKER_PORT,
                 client_id: str = MQTT_CLIENT_ID):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        
        self.client: Optional[mqtt.Client] = None
        self.is_connected = False
        self.lock = threading.Lock() # Đảm bảo an toàn khi chạy đa luồng
        
        # Biến cơ chế Edge-Trigger
        self.last_gesture = None        # Lưu tên cử chỉ cuối cùng đã gửi thành công
        self.last_gesture_time = 0.0    # Lưu thời điểm gửi cử chỉ cuối cùng
        
        # Thiết lập mDNS để ESP32 tự tìm thấy máy tính
        self.zc = Zeroconf()
        self._advertise_service()
        
        self._setup_client()
    
    def _advertise_service(self):
        # Quảng bá dịch vụ MQTT qua mDNS để ESP32 tự tìm thấy IP máy tính
        try:
            desc = {'version': '1.0'}
            # Lấy IP local của máy tính
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            info = ServiceInfo(
                "_mqtt._tcp.local.",
                "Mosquitto Broker._mqtt._tcp.local.",
                addresses=[socket.inet_aton(local_ip)],
                port=MQTT_BROKER_PORT,
                properties=desc,
                server="mosquitto.local.",
            )
            self.zc.register_service(info)
            print(f"mDNS: Đang quảng bá dịch vụ MQTT tại {local_ip}:{MQTT_BROKER_PORT}")
        except Exception as e:
            print(f"mDNS Advertisement failed: {e}")

    def _setup_client(self):
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.is_connected = True
            print(f"MQTT: Đã kết nối thành công tới Broker ({self.broker_host})")
        else:
            print(f"MQTT: Kết nối thất bại (Mã lỗi: {rc})")
    
    def _on_disconnect(self, client, userdata, rc):
        self.is_connected = False
        print(f"MQTT: Mất kết nối! Đang tự động thử lại...")

    def connect(self, timeout: float = 5.0) -> bool:
        try:
            self.client.connect(self.broker_host, self.broker_port, MQTT_KEEPALIVE)
            self.client.loop_start()
            
            # Đợi phản hồi kết nối trong khoảng thời gian timeout
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            return self.is_connected
        except Exception as e:
            print(f"MQTT: Lỗi trong quá trình kết nối: {e}")
            return False
    
    def disconnect(self):
        if self.zc:
            self.zc.unregister_all_services()
            self.zc.close()
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            print("MQTT: Đã ngắt kết nối.")
    
    def publish_gesture(self, 
                       gesture_label: str, 
                       confidence: float = 0.0,
                       hand_id: Optional[int] = None,
                       handedness: Optional[str] = None,
                       force: bool = False) -> bool:
        """
        CƠ CHẾ:
        1. Edge-Trigger: Với lệnh Bật/Tắt, chỉ gửi 1 lần khi đổi cử chỉ.
        2. Hold-to-Repeat: Với lệnh điều hướng quạt, cho phép gửi lại sau nếu giữ tay.
        """
        if not self.is_connected:
            return False
        
        current_time = time.time()
        
        # KIỂM TRA LẶP LỆNH (Trùng với lệnh vừa gửi)
        if self.last_gesture == gesture_label:
            # Nếu là lệnh điều hướng (Repeatable), kiểm tra cooldown lặp lại
            if gesture_label in REPEATABLE_GESTURES:
                if (current_time - self.last_gesture_time) < REPEAT_COOLDOWN_SECONDS:
                    return False
                # Nếu đã qua cooldown, cho phép gửi tiếp (Hold-to-Repeat)
            else:
                # Nếu KHÔNG phải lệnh điều hướng và không ép buộc (force), chặn gửi trùng
                if not force:
                    return False
                # Nếu ép buộc gửi trùng, vẫn phải cách nhau 1s
                if (current_time - self.last_gesture_time) < GESTURE_COOLDOWN_SECONDS:
                    return False
        
        # Đóng gói dữ liệu thành định dạng JSON để ESP32 dễ xử lý
        payload_dict = {
            "gesture": gesture_label,      # Tên lệnh (Vd: FanSpeed1)
            "confidence": round(confidence, 2), # Độ tin cậy (%)
            "timestamp": current_time,     # Thời gian gửi
            "hand_id": hand_id,            # ID của bàn tay
            "handedness": handedness       # Tay trái hay tay phải
        }
        
        try:
            payload = json.dumps(payload_dict, ensure_ascii=False)
            with self.lock: # Tránh xung đột dữ liệu khi nhiều tay gửi cùng lúc
                result = self.client.publish(MQTT_TOPIC_GESTURE, payload, qos=MQTT_QOS)
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    self.last_gesture = gesture_label
                    self.last_gesture_time = current_time
                    print(f"MQTT -> Gửi lệnh: {gesture_label} ({confidence:.1f}%)")
                    return True
                return False
        except Exception as e:
            print(f"MQTT: Lỗi khi phát tín hiệu: {e}")
            return False

# 2. CÁC HÀM
_mqtt_publisher: Optional[MQTTPublisher] = None

def init_mqtt_publisher() -> MQTTPublisher:
    global _mqtt_publisher
    if _mqtt_publisher is None:
        _mqtt_publisher = MQTTPublisher()
        _mqtt_publisher.connect()
    return _mqtt_publisher

def get_mqtt_publisher() -> Optional[MQTTPublisher]:
    return _mqtt_publisher

def publish_gesture_command(gesture_label: str, **kwargs) -> bool:
    pub = get_mqtt_publisher()
    if pub:
        return pub.publish_gesture(gesture_label, **kwargs)
    return False
