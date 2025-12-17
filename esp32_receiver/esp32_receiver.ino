#include <ESP32Servo.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// L298N #1 - Điều khiển 2 LED
#define LED_IN1 14
#define LED_IN2 27
#define LED_IN3 26
#define LED_IN4 25

// Buzzer GPIO (CHỌN CHÂN KHÁC VỚI L298 / SERVO / CAMERA)
// Gợi ý: dùng GPIO 4 (thường trống, không xung đột với L298 + servo hiện tại)
const int BUZZER_PIN = 4;

// L298N #2 - Điều khiển Quạt
const int FAN_ENA = 18;
const int FAN_IN1 = 32;
const int FAN_IN2 = 33;

// PWM config cho quạt - Không cần nữa vì dùng analogWrite
// const int freq = 5000;
// const int resolution = 8;

// Thời gian ramp cho quạt
const unsigned long rampMillis = 800;
const int rampSteps = 25;

// Trạng thái tốc độ quạt hiện tại
static int currentFanPercent = 0;

// Servo Motors - ESP32Servo sẽ tự động assign channels
Servo servo1;
Servo servo2;
const int servo1Pin = 16;    // GPIO 16 cho servo 1
const int servo2Pin = 17;    // GPIO 17 cho servo 2
int servo1CurrentAngle = 90; // Vị trí hiện tại của servo 1
int servo2CurrentAngle = 90; // Vị trí hiện tại của servo 2

// ========= WiFi & MQTT CONFIG (CHỈ TEST ĐÈN, GIỮ LẠI CODE QUẠT + SERVO ĐỂ SAU) =========
const char *ssid = "ilovehcmute";  // Tên WiFi
const char *password = "910JQKA2"; // Mật khẩu WiFi

// MQTT Broker settings (PC bạn)
const char *mqtt_broker = "192.168.115.253";
const int mqtt_port = 1883;
const char *mqtt_client_id = "esp32_light_receiver_test";

// Topics
const char *mqtt_topic_gesture = "gesture/command";

// WiFi & MQTT client
WiFiClient espClient;
PubSubClient mqtt_client(espClient);

// State demo cho LED (để tránh lệnh trùng)
bool light1_state = false;
bool light2_state = false;
int fan_speed_state = 0; // 0 = OFF, 1/2/3 tương ứng FanSpeed1/2/3

// ================= BUZZER HELPERS (dùng cho Start + action beep) =================

// Khởi tạo PWM cho buzzer (ESP32 Arduino Core 3.x)
void initBuzzer()
{
  // pin, freq, resolution
  ledcAttach(BUZZER_PIN, 2000, 8); // 2kHz, 8-bit
}

void buzzerTone(int freq, int durationMs)
{
  if (freq <= 0)
  {
    ledcWriteTone(BUZZER_PIN, 0);
    delay(durationMs);
    return;
  }
  ledcWriteTone(BUZZER_PIN, freq);
  delay(durationMs);
  ledcWriteTone(BUZZER_PIN, 0);
}

// Beep ngắn cho mỗi hành động điều khiển (ON/OFF, đổi tốc quạt, ...)
void buzzerActionBeep()
{
  buzzerTone(1200, 80); // 1 nốt ngắn
}

// Beep 2 nốt khi nhận gesture "Start"
void buzzerStartBeep()
{
  buzzerTone(1000, 120);
  delay(40);
  buzzerTone(1600, 160);
}

// Khai báo hàm MQTT/WiFi trước
void connectWiFi();
void connectMQTT();
void mqtt_callback(char *topic, byte *payload, unsigned int length);
void parse_gesture_message(const char *json_string);

void setup()
{
  Serial.begin(115200);
  delay(100);

  Serial.println();
  Serial.println("============================================================");
  Serial.println("ESP32 MQTT Light Receiver - (LED + Fan + Servo hardware ready)");
  Serial.println("  → Hiện tại CHỈ dùng MQTT để bật/tắt LED 1 & 2");
  Serial.println("  → Code quạt + servo vẫn GIỮ LẠI để xử lý sau");
  Serial.println("============================================================");

  pinMode(LED_IN1, OUTPUT);
  pinMode(LED_IN2, OUTPUT);
  pinMode(LED_IN3, OUTPUT);
  pinMode(LED_IN4, OUTPUT);

  digitalWrite(LED_IN1, LOW);
  digitalWrite(LED_IN2, LOW);
  digitalWrite(LED_IN3, LOW);
  digitalWrite(LED_IN4, LOW);

  pinMode(FAN_IN1, OUTPUT);
  pinMode(FAN_IN2, OUTPUT);
  digitalWrite(FAN_IN1, HIGH);
  digitalWrite(FAN_IN2, LOW);

  // KHỞI TẠO SERVO TRƯỚC - Để ESP32Servo chiếm channels riêng
  Serial.println("\n=== INITIALIZING SERVOS FIRST ===");
  servo1.attach(servo1Pin, 500, 2400);
  servo2.attach(servo2Pin, 500, 2400);
  Serial.print("Servo1 attached to pin ");
  Serial.println(servo1Pin);
  Serial.print("Servo2 attached to pin ");
  Serial.println(servo2Pin);
  Serial.println("Servos will use their own PWM channels");
  delay(200);

  // KHỞI TẠO QUẠT SAU - Dùng analogWrite để tránh xung đột
  Serial.println("\n=== INITIALIZING FAN ===");
  pinMode(FAN_ENA, OUTPUT);
  analogWrite(FAN_ENA, 0);
  Serial.println("Fan PWM configured using analogWrite (no conflict)");
  setFanSpeed(0);
  delay(100);
  Serial.print("Servo1 attached to pin ");
  Serial.println(servo1Pin);
  Serial.print("Servo2 attached to pin ");
  Serial.println(servo2Pin);
  Serial.println("(Channels auto-assigned by ESP32Servo library)");

  // (OPTIONAL) Test servo nhanh - CÓ THỂ BỎ QUA nếu không cần test lúc khởi động
  // Giữ code lại để sau này debug, nhưng hiện tại không cần chạy liên tục.
  /*
  Serial.println("Testing servos...");
  servo1.write(90); // Vị trí giữa
  servo2.write(90); // Vị trí giữa
  servo1CurrentAngle = 90;
  servo2CurrentAngle = 90;
  delay(500);

  Serial.println("\n=== TESTING SERVO 1 (GPIO 16) ===");
  Serial.println("Servo1 -> 0 degrees");
  servo1.write(0);
  delay(500);
  Serial.println("Servo1 -> 180 degrees");
  servo1.write(180);
  delay(500);
  Serial.println("Servo1 -> 90 degrees (center)");
  servo1.write(90);
  delay(500);

  Serial.println("\n=== TESTING SERVO 2 (GPIO 17) ===");
  Serial.println("Servo2 -> 0 degrees");
  servo2.write(0);
  delay(500);
  Serial.println("Servo2 -> 180 degrees");
  servo2.write(180);
  delay(500);
  Serial.println("Servo2 -> 90 degrees (center)");
  servo2.write(90);
  delay(500);

  Serial.println("\n=== SERVO TEST COMPLETE ===");
  */

  Serial.println("Setup complete! (LED + Fan + 2 Servos hardware ready)");
  delay(500);

  // ========= BUZZER =========
  initBuzzer();

  // ========= KẾT NỐI WIFI + MQTT (CHO TEST ĐÈN) =========
  connectWiFi();
  mqtt_client.setServer(mqtt_broker, mqtt_port);
  mqtt_client.setCallback(mqtt_callback);
  connectMQTT();
}

// Hàm điều khiển LED
void led1On()
{
  digitalWrite(LED_IN1, HIGH);
  digitalWrite(LED_IN2, LOW);
  Serial.println("LED 1: ON");
}

void led1Off()
{
  digitalWrite(LED_IN1, LOW);
  digitalWrite(LED_IN2, LOW);
  Serial.println("LED 1: OFF");
}

void led2On()
{
  digitalWrite(LED_IN3, HIGH);
  digitalWrite(LED_IN4, LOW);
  Serial.println("LED 2: ON");
}

void led2Off()
{
  digitalWrite(LED_IN3, LOW);
  digitalWrite(LED_IN4, LOW);
  Serial.println("LED 2: OFF");
}

void allLedOff()
{
  led1Off();
  led2Off();
}

// Hàm điều khiển Quạt
void setFanSpeed(int percent)
{
  if (percent < 0)
    percent = 0;
  if (percent > 100)
    percent = 100;
  int pwmVal = map(percent, 0, 100, 0, 255);

  // Dùng analogWrite - không xung đột với ESP32Servo
  analogWrite(FAN_ENA, pwmVal);
}

void rampFanTo(int targetPercent)
{
  if (targetPercent < 0)
    targetPercent = 0;
  if (targetPercent > 100)
    targetPercent = 100;

  int start = currentFanPercent;
  int end = targetPercent;

  if (start == end)
    return;

  // Kick-start: Nếu quạt đang tắt và muốn bật, khởi động ở 100% trước
  if (start == 0 && end > 0)
  {
    Serial.println("Kick-start: 100% for 500ms");
    setFanSpeed(100);
    delay(500);
    start = 100;
  }

  for (int step = 1; step <= rampSteps; ++step)
  {
    float t = (float)step / (float)rampSteps;
    int now = start + (int)((end - start) * t);
    setFanSpeed(now);
    delay(rampMillis / rampSteps);
  }

  setFanSpeed(end);
  currentFanPercent = end;
  Serial.print("Fan speed: ");
  Serial.print(end);
  Serial.println("%");
}

// Hàm điều khiển Servo
void setServo1Angle(int angle)
{
  if (angle < 0)
    angle = 0;
  if (angle > 180)
    angle = 180;
  servo1.attach(servo1Pin); // Đảm bảo servo được attach
  servo1.write(angle);
  servo1CurrentAngle = angle;
  Serial.print("Servo 1: ");
  Serial.print(angle);
  Serial.println(" degrees");
}

void setServo2Angle(int angle)
{
  if (angle < 0)
    angle = 0;
  if (angle > 180)
    angle = 180;
  servo2.attach(servo2Pin); // Đảm bảo servo được attach
  servo2.write(angle);
  servo2CurrentAngle = angle;
  Serial.print("Servo 2: ");
  Serial.print(angle);
  Serial.println(" degrees");
}

void setBothServos(int angle1, int angle2)
{
  setServo1Angle(angle1);
  setServo2Angle(angle2);
}

void moveServo1Smooth(int targetAngle, int stepDelay = 30)
{
  if (targetAngle < 0)
    targetAngle = 0;
  if (targetAngle > 180)
    targetAngle = 180;

  // Đảm bảo servo được attach
  if (!servo1.attached())
  {
    servo1.attach(servo1Pin, 500, 2400);
  }
  int currentAngle = servo1CurrentAngle;

  if (currentAngle < targetAngle)
  {
    for (int pos = currentAngle; pos <= targetAngle; pos += 1)
    {
      servo1.write(pos);
      delay(stepDelay);
    }
  }
  else if (currentAngle > targetAngle)
  {
    for (int pos = currentAngle; pos >= targetAngle; pos -= 1)
    {
      servo1.write(pos);
      delay(stepDelay);
    }
  }
  servo1CurrentAngle = targetAngle;
  Serial.print("Servo 1 moved to: ");
  Serial.println(targetAngle);
}

void moveServo2Smooth(int targetAngle, int stepDelay = 30)
{
  if (targetAngle < 0)
    targetAngle = 0;
  if (targetAngle > 180)
    targetAngle = 180;

  // Đảm bảo servo được attach
  if (!servo2.attached())
  {
    servo2.attach(servo2Pin, 500, 2400);
  }
  int currentAngle = servo2CurrentAngle;

  if (currentAngle < targetAngle)
  {
    for (int pos = currentAngle; pos <= targetAngle; pos += 1)
    {
      servo2.write(pos);
      delay(stepDelay);
    }
  }
  else if (currentAngle > targetAngle)
  {
    for (int pos = currentAngle; pos >= targetAngle; pos -= 1)
    {
      servo2.write(pos);
      delay(stepDelay);
    }
  }
  servo2CurrentAngle = targetAngle;
  Serial.print("Servo 2 moved to: ");
  Serial.println(targetAngle);
}

// ================= LOOP MỚI: CHỈ XỬ LÝ MQTT (GIỮ CODE DEMO CŨ, NHƯNG COMMENT) =================

// LOOP dùng cho MQTT: duy trì kết nối và nhận lệnh bật/tắt đèn
void loop()
{
  if (!mqtt_client.connected())
  {
    connectMQTT();
  }
  else
  {
    mqtt_client.loop();
  }

  delay(50); // Nhỏ thôi cho nhẹ CPU
}

// --- GIỮ NGUYÊN CODE DEMO CŨ, NHƯNG KHÔNG COMPILE / KHÔNG CHẠY (ĐỂ SAU NÀY XỬ LÝ QUẠT + SERVO) ---
/*
// LOOP - Demo điều khiển (LED + Fan + 2 Servos)
void demoLoop()
{
  // Demo 1: Bật LED 1, quạt 50%, Servo 1 ở 0 độ
  Serial.println("\n--- LED1 ON + Fan 50% + Servo1(0) ---");
  led1On();
  rampFanTo(50);
  moveServo1Smooth(0);
  delay(5000);

  // Demo 2: Bật LED 2, quạt 75%, Servo 2 ở 90 độ
  Serial.println("\n--- LED2 ON + Fan 75% + Servo2(90) ---");
  led1Off();
  led2On();
  rampFanTo(75);
  moveServo2Smooth(90);
  delay(5000);

  // Demo 3: Cả 2 LED sáng, quạt 100%, cả 2 servo di chuyển
  Serial.println("\n--- Both LED ON + Fan 100% + Both Servos ---");
  led1On();
  led2On();
  rampFanTo(100);
  // Di chuyển cả 2 servo cùng lúc
  servo1.attach(servo1Pin);
  servo2.attach(servo2Pin);
  for (int pos = 0; pos <= 180; pos += 1)
  {
    servo1.write(pos);
    servo1CurrentAngle = pos;
    if (pos <= 90)
    {
      int servo2Pos = 90 + pos;
      servo2.write(servo2Pos);
      servo2CurrentAngle = servo2Pos;
    }
    delay(30);
  }
  servo1CurrentAngle = 180;
  servo2CurrentAngle = 180;
  servo1.detach();
  servo2.detach();
  delay(2000);

  // Demo 4: Servo di chuyển ngược chiều, quạt giảm
  Serial.println("\n--- Servos Opposite + Fan 50% ---");
  rampFanTo(50);
  if (!servo1.attached())
  {
    servo1.attach(servo1Pin, 500, 2400);
  }
  if (!servo2.attached())
  {
    servo2.attach(servo2Pin, 500, 2400);
  }
  for (int pos = 180; pos >= 0; pos -= 1)
  {
    servo1.write(pos);
    servo1CurrentAngle = pos;
    int servo2Pos = 180 - pos;
    servo2.write(servo2Pos);
    servo2CurrentAngle = servo2Pos;
    delay(30);
  }
  servo1CurrentAngle = 0;
  servo2CurrentAngle = 180;
  servo1.detach();
  servo2.detach();
  delay(2000);

  // Demo 5: Tắt hết, servo về vị trí giữa
  Serial.println("\n--- All OFF + Servos to Center ---");
  allLedOff();
  rampFanTo(0);
  moveServo1Smooth(90);
  moveServo2Smooth(90);
  delay(5000);
}
*/

// ================= WIFI + MQTT IMPLEMENTATION (CHO TEST BẬT/TẮT ĐÈN) =================

void connectWiFi()
{
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30)
  {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED)
  {
    Serial.println("✓ WiFi connected!");
    Serial.print("  IP address: ");
    Serial.println(WiFi.localIP());
  }
  else
  {
    Serial.println("✗ WiFi connection failed!");
  }
}

void connectMQTT()
{
  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println("⚠ WiFi not connected, cannot connect to MQTT");
    return;
  }

  Serial.print("Connecting to MQTT broker: ");
  Serial.print(mqtt_broker);
  Serial.print(":");
  Serial.println(mqtt_port);

  if (mqtt_client.connect(mqtt_client_id))
  {
    Serial.println("✓ MQTT connected!");

    if (mqtt_client.subscribe(mqtt_topic_gesture))
    {
      Serial.print("  ✓ Subscribed to topic: ");
      Serial.println(mqtt_topic_gesture);
    }
    else
    {
      Serial.println("  ✗ Failed to subscribe to topic");
    }
  }
  else
  {
    Serial.print("✗ MQTT connection failed, rc=");
    Serial.println(mqtt_client.state());
    Serial.println("  → Check if MQTT broker is running on PC");
    Serial.print("  → Broker IP should be: ");
    Serial.println(mqtt_broker);
  }
}

void mqtt_callback(char *topic, byte *payload, unsigned int length)
{
  // Convert payload to string
  char message[length + 1];
  memcpy(message, payload, length);
  message[length] = '\0';

  Serial.println();
  Serial.println("============================================================");
  Serial.println("📨 MQTT Message Received");
  Serial.println("============================================================");
  Serial.print("Topic: ");
  Serial.println(topic);
  Serial.print("Payload length: ");
  Serial.println(length);
  Serial.print("Raw payload: ");
  Serial.println(message);
  Serial.println();

  if (strcmp(topic, mqtt_topic_gesture) == 0)
  {
    parse_gesture_message(message);
  }
  else
  {
    Serial.println("Unknown topic, ignore");
  }

  Serial.println("============================================================");
  Serial.println();
}

void parse_gesture_message(const char *json_string)
{
  Serial.println("📋 Parsing Gesture Command:");
  Serial.println("------------------------------------------------------------");

  StaticJsonDocument<256> doc;
  DeserializationError error = deserializeJson(doc, json_string);

  if (error)
  {
    Serial.print("✗ JSON parse error: ");
    Serial.println(error.c_str());
    Serial.println("Raw message:");
    Serial.println(json_string);
    return;
  }

  const char *gesture = doc["gesture"] | "UNKNOWN";
  float confidence = doc["confidence"] | 0.0;

  Serial.println("Parsed Data:");
  Serial.print("  Gesture: ");
  Serial.println(gesture);
  Serial.print("  Confidence: ");
  Serial.print(confidence);
  Serial.println("%");

  Serial.println();

  // Gesture "Start": beep 2 nốt để báo hệ thống sẵn sàng
  if (strcmp(gesture, "Start") == 0)
  {
    Serial.println("💡 Action: START gesture detected → buzzer 2 notes");
    buzzerStartBeep();
    return;
  }

  Serial.println("💡 Action (LED + Fan Test):");

  // ================= ĐÈN: Light1On / Light1Off / Light2On / Light2Off =================
  if (strcmp(gesture, "Light1On") == 0)
  {
    if (!light1_state)
    {
      light1_state = true;
      led1On();
      buzzerActionBeep();
      Serial.println("  → LIGHT 1: ON (state OFF → ON)");
    }
    else
    {
      Serial.println("  → LIGHT 1: ĐÃ ON sẵn, bỏ qua lệnh trùng");
    }
  }
  else if (strcmp(gesture, "Light1Off") == 0)
  {
    if (light1_state)
    {
      light1_state = false;
      led1Off();
      buzzerActionBeep();
      Serial.println("  → LIGHT 1: OFF (state ON → OFF)");
    }
    else
    {
      Serial.println("  → LIGHT 1: ĐÃ OFF sẵn, bỏ qua lệnh trùng");
    }
  }
  else if (strcmp(gesture, "Light2On") == 0)
  {
    if (!light2_state)
    {
      light2_state = true;
      led2On();
      buzzerActionBeep();
      Serial.println("  → LIGHT 2: ON (state OFF → ON)");
    }
    else
    {
      Serial.println("  → LIGHT 2: ĐÃ ON sẵn, bỏ qua lệnh trùng");
    }
  }
  else if (strcmp(gesture, "Light2Off") == 0)
  {
    if (light2_state)
    {
      light2_state = false;
      led2Off();
      buzzerActionBeep();
      Serial.println("  → LIGHT 2: OFF (state ON → OFF)");
    }
    else
    {
      Serial.println("  → LIGHT 2: ĐÃ OFF sẵn, bỏ qua lệnh trùng");
    }
  }
  // ================= QUẠT: FanOff / FanSpeed1 / FanSpeed2 / FanSpeed3 =================
  else if (strcmp(gesture, "FanOff") == 0)
  {
    if (fan_speed_state != 0)
    {
      fan_speed_state = 0;
      rampFanTo(0); // Tắt quạt (ramp về 0%)
      buzzerActionBeep();
      Serial.println("  → FAN: OFF (state >0 → 0%, tắt quạt)");
    }
    else
    {
      Serial.println("  → FAN: đã OFF sẵn, bỏ qua lệnh trùng");
    }
  }
  else if (strcmp(gesture, "FanSpeed1") == 0)
  {
    if (fan_speed_state != 1)
    {
      fan_speed_state = 1;
      rampFanTo(60); // 60%
      buzzerActionBeep();
      Serial.println("  → FAN: set speed = 60% (FanSpeed1)");
    }
    else
    {
      Serial.println("  → FAN: speed 60% (FanSpeed1) đã được set sẵn, bỏ qua lệnh trùng");
    }
  }
  else if (strcmp(gesture, "FanSpeed2") == 0)
  {
    if (fan_speed_state != 2)
    {
      fan_speed_state = 2;
      rampFanTo(80); // 80%
      buzzerActionBeep();
      Serial.println("  → FAN: set speed = 80% (FanSpeed2)");
    }
    else
    {
      Serial.println("  → FAN: speed 80% (FanSpeed2) đã được set sẵn, bỏ qua lệnh trùng");
    }
  }
  else if (strcmp(gesture, "FanSpeed3") == 0)
  {
    if (fan_speed_state != 3)
    {
      fan_speed_state = 3;
      rampFanTo(100); // 100%
      buzzerActionBeep();
      Serial.println("  → FAN: set speed = 100% (FanSpeed3)");
    }
    else
    {
      Serial.println("  → FAN: speed 100% (FanSpeed3) đã được set sẵn, bỏ qua lệnh trùng");
    }
  }
  else
  {
    Serial.print("  → Unknown gesture for this test: ");
    Serial.println(gesture);
  }
}