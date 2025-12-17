import paho.mqtt.client as mqtt
import json
import time
import threading
from typing import Optional, Callable

# ========== CONFIGURATION ==========
# MQTT Broker settings (mặc định: localhost - chạy trên cùng PC)
MQTT_BROKER_HOST = "localhost"  # Hoặc IP của MQTT broker (ví dụ: "192.168.1.2")
MQTT_BROKER_PORT = 1883  # Port mặc định của MQTT
MQTT_KEEPALIVE = 60  # Keepalive interval (giây)

# MQTT Topics
MQTT_TOPIC_GESTURE = "gesture/command"  # Topic để publish lệnh gesture
MQTT_TOPIC_STATUS = "gesture/status"    # Topic để publish status (optional)

# Client settings
MQTT_CLIENT_ID = "gesture_detector_publisher"  # Unique client ID
MQTT_QOS = 1  # Quality of Service: 0 (at most once), 1 (at least once), 2 (exactly once)

# Reconnection settings
MQTT_RECONNECT_DELAY = 5  # Thời gian chờ trước khi reconnect (giây)
MQTT_MAX_RECONNECT_ATTEMPTS = 10  # Số lần thử reconnect tối đa (0 = vô hạn)

# Gesture cooldown (tránh spam MQTT khi gesture giữ lâu)
# LƯU Ý: Với cơ chế edge-trigger bên dưới, cooldown theo thời gian gần như không cần,
# nhưng vẫn giữ lại để phòng trường hợp muốn bật lại cùng gesture bằng force=True.
GESTURE_COOLDOWN_SECONDS = 0.5  # Chỉ dùng khi force=True hoặc khi muốn cho phép gửi lại cùng gesture
# ===================================


class MQTTPublisher:
    def __init__(self, 
                 broker_host: str = MQTT_BROKER_HOST,
                 broker_port: int = MQTT_BROKER_PORT,
                 client_id: str = MQTT_CLIENT_ID,
                 on_connect_callback: Optional[Callable] = None,
                 on_disconnect_callback: Optional[Callable] = None):
        """
            broker_host: IP hoặc hostname của MQTT broker
            broker_port: Port của MQTT broker (mặc định 1883)
            client_id: Unique client ID cho MQTT client
            on_connect_callback: Callback khi kết nối thành công (optional)
            on_disconnect_callback: Callback khi mất kết nối (optional)
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.on_connect_callback = on_connect_callback
        self.on_disconnect_callback = on_disconnect_callback
        
        # MQTT client
        self.client: Optional[mqtt.Client] = None
        self.is_connected = False
        self.reconnect_attempts = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Gesture cooldown / edge-trigger tracking
        self.last_gesture = None
        self.last_gesture_time = 0.0
        
        # Callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
    
    def _on_connect(self, client, userdata, flags, rc):
        """
        Callback khi kết nối MQTT broker
        
        rc (return code):
            0: Connection successful
            1: Connection refused - incorrect protocol version
            2: Connection refused - invalid client identifier
            3: Connection refused - server unavailable
            4: Connection refused - bad username or password
            5: Connection refused - not authorised
        """
        if rc == 0:
            self.is_connected = True
            self.reconnect_attempts = 0
            print(f"✓ MQTT Connected to {self.broker_host}:{self.broker_port}")
            if self.on_connect_callback:
                self.on_connect_callback()
        else:
            self.is_connected = False
            error_messages = {
                1: "incorrect protocol version",
                2: "invalid client identifier",
                3: "server unavailable",
                4: "bad username or password",
                5: "not authorised"
            }
            error_msg = error_messages.get(rc, f"unknown error (code {rc})")
            print(f"✗ MQTT Connection failed: {error_msg}")
    
    def _on_disconnect(self, client, userdata, rc):
        self.is_connected = False
        print(f"⚠ MQTT Disconnected from {self.broker_host}:{self.broker_port}")
        if self.on_disconnect_callback:
            self.on_disconnect_callback()
    
    def _on_publish(self, client, userdata, mid):
        """Callback khi publish thành công (optional, để debug)"""
        # Có thể log ở đây nếu cần debug
        pass
    
    def connect(self, timeout: float = 5.0) -> bool:
        """
        Kết nối đến MQTT broker
        
        Args:
            timeout: Thời gian timeout cho kết nối (giây)
        
        Returns:
            True nếu kết nối thành công, False nếu thất bại
        """
        if self.client is None:
            self._setup_callbacks()
        
        try:
            print(f"→ Connecting to MQTT broker: {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, MQTT_KEEPALIVE)
            self.client.loop_start()  # Start network loop in background thread
            
            # Đợi kết nối (với timeout)
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.is_connected:
                return True
            else:
                print(f"✗ MQTT Connection timeout after {timeout}s")
                return False
                
        except Exception as e:
            print(f"✗ MQTT Connection error: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Ngắt kết nối MQTT broker"""
        if self.client and self.is_connected:
            try:
                self.client.loop_stop()
                self.client.disconnect()
                self.is_connected = False
                print("✓ MQTT Disconnected")
            except Exception as e:
                print(f"⚠ MQTT Disconnect error: {e}")
    
    def publish_gesture(self, 
                       gesture_label: str, 
                       confidence: float = 0.0,
                       hand_id: Optional[int] = None,
                       handedness: Optional[str] = None,
                       force: bool = False) -> bool:
        """
        Publish lệnh gesture qua MQTT.

        CƠ CHẾ MỚI (EDGE-TRIGGER):
        - Chỉ publish khi gesture THAY ĐỔI so với lần gần nhất.
        - Nếu cùng gesture_label như lần trước thì bỏ qua (tránh spam khi người dùng giữ tay lâu).
        - Nếu muốn publish lại cùng gesture (ví dụ cho mục đích debug / reset), truyền force=True.
        
        Args:
            gesture_label: Tên gesture (ví dụ: "FanSpeed1", "FanSpeed2", "FanLeft", v.v.)
            confidence: Confidence score (0.0 - 100.0)
            hand_id: ID của hand (optional)
            handedness: "Left" hoặc "Right" (optional)
            force: Bỏ qua edge-trigger (và cooldown) nếu True (default: False)
        
        Returns:
            True nếu publish thành công, False nếu thất bại
        """
        if not self.is_connected:
            print("⚠ MQTT not connected, cannot publish gesture")
            return False
        
        # EDGE-TRIGGER: chỉ gửi khi gesture thay đổi
        if not force and self.last_gesture == gesture_label:
            # Cùng gesture như lần trước → không gửi lại, tránh spam khi giữ tay lâu
            return False

        # OPTIONAL: nếu muốn vẫn giới hạn tần suất với force=True,
        # có thể dùng thêm cooldown theo thời gian bên dưới.
        current_time = time.time()
        if force and self.last_gesture == gesture_label:
            time_since_last = current_time - self.last_gesture_time
            if time_since_last < GESTURE_COOLDOWN_SECONDS:
                # force=True nhưng vẫn đang trong cooldown → bỏ qua để tránh spam cứng
                return False
        
        # Tạo message payload (JSON format)
        message = {
            "gesture": gesture_label,
            "confidence": round(confidence, 2),
            "timestamp": current_time,
            "hand_id": hand_id,
            "handedness": handedness
        }
        
        # Convert to JSON string
        try:
            payload = json.dumps(message, ensure_ascii=False)
        except Exception as e:
            print(f"✗ Error encoding JSON: {e}")
            return False
        
        # Publish message
        try:
            with self.lock:
                result = self.client.publish(
                    MQTT_TOPIC_GESTURE,
                    payload,
                    qos=MQTT_QOS
                )
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    # Update cooldown tracking
                    self.last_gesture = gesture_label
                    self.last_gesture_time = current_time
                    print(f"✓ MQTT Published: {gesture_label} (conf: {confidence:.1f}%)")
                    return True
                else:
                    print(f"✗ MQTT Publish failed: error code {result.rc}")
                    return False
                    
        except Exception as e:
            print(f"✗ MQTT Publish error: {e}")
            return False
    
    def publish_status(self, status: str, message: Optional[str] = None) -> bool:
        """
        Publish status message (optional, để debug/monitoring)
        
        Args:
            status: Status string (ví dụ: "detecting", "idle", "error")
            message: Optional message details
        
        Returns:
            True nếu publish thành công, False nếu thất bại
        """
        if not self.is_connected:
            return False
        
        payload = {
            "status": status,
            "message": message,
            "timestamp": time.time()
        }
        
        try:
            payload_str = json.dumps(payload, ensure_ascii=False)
            result = self.client.publish(
                MQTT_TOPIC_STATUS,
                payload_str,
                qos=MQTT_QOS
            )
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            print(f"✗ MQTT Status publish error: {e}")
            return False


# ========== Global MQTT Publisher Instance ==========
# Tạo global instance để dùng trong các file khác
_mqtt_publisher: Optional[MQTTPublisher] = None


def init_mqtt_publisher(broker_host: str = MQTT_BROKER_HOST,
                       broker_port: int = MQTT_BROKER_PORT,
                       client_id: str = MQTT_CLIENT_ID) -> MQTTPublisher:
    """
    Khởi tạo global MQTT publisher instance
    
    Args:
        broker_host: IP hoặc hostname của MQTT broker
        broker_port: Port của MQTT broker
        client_id: Unique client ID
    
    Returns:
        MQTTPublisher instance
    """
    global _mqtt_publisher
    
    if _mqtt_publisher is None:
        _mqtt_publisher = MQTTPublisher(
            broker_host=broker_host,
            broker_port=broker_port,
            client_id=client_id
        )
        # Tự động kết nối
        _mqtt_publisher.connect()
    
    return _mqtt_publisher


def get_mqtt_publisher() -> Optional[MQTTPublisher]:
    """Lấy global MQTT publisher instance (nếu đã khởi tạo)"""
    return _mqtt_publisher


def publish_gesture_command(gesture_label: str,
                           confidence: float = 0.0,
                           hand_id: Optional[int] = None,
                           handedness: Optional[str] = None,
                           force: bool = False) -> bool:
    """
    Helper function để publish gesture command (dùng global instance)
    
    Args:
        gesture_label: Tên gesture
        confidence: Confidence score (0.0 - 100.0)
        hand_id: ID của hand (optional)
        handedness: "Left" hoặc "Right" (optional)
        force: Bỏ qua cooldown nếu True
    
    Returns:
        True nếu publish thành công, False nếu thất bại hoặc chưa init
    """
    global _mqtt_publisher
    
    if _mqtt_publisher is None:
        print("⚠ MQTT Publisher chưa được khởi tạo. Gọi init_mqtt_publisher() trước.")
        return False
    
    return _mqtt_publisher.publish_gesture(
        gesture_label=gesture_label,
        confidence=confidence,
        hand_id=hand_id,
        handedness=handedness,
        force=force
    )


# ========== Example Usage ==========
if __name__ == "__main__":
    # Test MQTT publisher
    print("=" * 60)
    print("MQTT Publisher Test")
    print("=" * 60)
    
    # Khởi tạo publisher
    publisher = init_mqtt_publisher()
    
    if publisher.is_connected:
        # Test publish một số gestures
        test_gestures = [
            ("FanSpeed1", 95.5, 0, "Right"),
            ("FanSpeed2", 92.3, 0, "Right"),
            ("FanSpeed3", 88.7, 0, "Right"),
        ]
        
        for gesture, conf, hand_id, handedness in test_gestures:
            publisher.publish_gesture(
                gesture_label=gesture,
                confidence=conf,
                hand_id=hand_id,
                handedness=handedness
            )
            time.sleep(1)  # Đợi 1s giữa các lần publish
        
        # Disconnect
        time.sleep(2)
        publisher.disconnect()
    else:
        print("✗ Không thể kết nối MQTT broker")
        print("  → Đảm bảo MQTT broker đang chạy (ví dụ: mosquitto)")
        print("  → Hoặc thay đổi MQTT_BROKER_HOST trong file này")

