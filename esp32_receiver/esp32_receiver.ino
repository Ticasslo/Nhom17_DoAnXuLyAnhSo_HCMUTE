#include <ESP32Servo.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// CONFIGURATION
const char *ssid = "ilovehcmute";
const char *password = "910JQKA2";
const int mqtt_port = 1883;
const char *mqtt_client_id = "esp32_receiver";
const char *mqtt_topic_gesture = "gesture/command";

// MQTT Broker Fallback IP (dùng nếu mDNS không tìm thấy broker)
// Để trống "" để chỉ dùng mDNS, hoặc nhập IP trực tiếp ví dụ: "192.168.115.253"
const char *mqtt_broker_fallback_ip = "192.168.115.253";

// PIN DEFINITIONS
#define LED_IN1 14
#define LED_IN2 27
#define LED_IN3 26
#define LED_IN4 25
const int BUZZER_PIN = 4;
const int FAN_ENA = 18;
const int FAN_IN1 = 32;
const int FAN_IN2 = 33;
const int servo1Pin = 16;
const int servo2Pin = 17;

// FAN CONFIG
const unsigned long rampMillis = 800;
const int rampSteps = 25;
static int currentFanPercent = 0;
int fan_speed_state = 0; // 0=OFF, 1/2/3=Speed

// SERVO CONFIG
Servo servo1;
int servo1CurrentAngle = 90;
int servo1Angles[] = {50, 90, 130};
int servo1AngleIndex = 1;
const int servo1MaxIndex = 2;

Servo servo2;
int servo2CurrentAngle = 35;
int servo2Angles[] = {0, 35, 70};
int servo2AngleIndex = 1;
const int servo2MaxIndex = 2;

// STATE TRACKING
bool light1_state = false;
bool light2_state = false;

// MQTT & WIFI CLIENTS
WiFiClient espClient;
PubSubClient mqtt_client(espClient);
String discovered_mqtt_broker = "";

// BUZZER HELPERS
void initBuzzer()
{
  ledcAttach(BUZZER_PIN, 2000, 8);
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

void buzzerActionBeep() { buzzerTone(1200, 80); }
void buzzerStartBeep()
{
  buzzerTone(1000, 120);
  delay(40);
  buzzerTone(1600, 160);
}

// HARDWARE CONTROL
void led1On()
{
  digitalWrite(LED_IN1, HIGH);
  digitalWrite(LED_IN2, LOW);
  Serial.println("  [HARDWARE] LED 1 set to HIGH/LOW (ON)");
}
void led1Off()
{
  digitalWrite(LED_IN1, LOW);
  digitalWrite(LED_IN2, LOW);
  Serial.println("  [HARDWARE] LED 1 set to LOW/LOW (OFF)");
}
void led2On()
{
  digitalWrite(LED_IN3, HIGH);
  digitalWrite(LED_IN4, LOW);
  Serial.println("  [HARDWARE] LED 2 set to HIGH/LOW (ON)");
}
void led2Off()
{
  digitalWrite(LED_IN3, LOW);
  digitalWrite(LED_IN4, LOW);
  Serial.println("  [HARDWARE] LED 2 set to LOW/LOW (OFF)");
}
void allLedOff()
{
  led1Off();
  led2Off();
}

void setFanSpeed(int percent)
{
  percent = constrain(percent, 0, 100);
  int pwmVal = map(percent, 0, 100, 0, 255);
  analogWrite(FAN_ENA, pwmVal);
}

void rampFanTo(int targetPercent)
{
  targetPercent = constrain(targetPercent, 0, 100);
  int start = currentFanPercent;
  if (start == targetPercent)
    return;

  if (start == 0 && targetPercent > 0)
  {
    setFanSpeed(100);
    delay(500);
    start = 100;
  }

  for (int step = 1; step <= rampSteps; ++step)
  {
    float t = (float)step / (float)rampSteps;
    int now = start + (int)((targetPercent - start) * t);
    setFanSpeed(now);
    delay(rampMillis / rampSteps);
  }
  setFanSpeed(targetPercent);
  currentFanPercent = targetPercent;
}

void moveServo1Smooth(int targetAngle, int stepDelay = 20)
{
  targetAngle = constrain(targetAngle, 0, 180);
  int startAngle = servo1CurrentAngle;
  if (startAngle == targetAngle)
    return;

  if (startAngle < targetAngle)
  {
    for (int pos = startAngle; pos <= targetAngle; pos++)
    {
      servo1.write(pos);
      delay(stepDelay);
    }
  }
  else
  {
    for (int pos = startAngle; pos >= targetAngle; pos--)
    {
      servo1.write(pos);
      delay(stepDelay);
    }
  }
  servo1CurrentAngle = targetAngle;
}

void moveServo2Smooth(int targetAngle, int stepDelay = 20)
{
  targetAngle = constrain(targetAngle, 0, 180);
  int startAngle = servo2CurrentAngle;
  if (startAngle == targetAngle)
    return;

  if (startAngle < targetAngle)
  {
    for (int pos = startAngle; pos <= targetAngle; pos++)
    {
      servo2.write(pos);
      delay(stepDelay);
    }
  }
  else
  {
    for (int pos = startAngle; pos >= targetAngle; pos--)
    {
      servo2.write(pos);
      delay(stepDelay);
    }
  }
  servo2CurrentAngle = targetAngle;
}

// NETWORK & MQTT
void connectWiFi()
{
  Serial.print("→ Connecting WiFi: ");
  Serial.println(ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20)
  {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED)
  {
    Serial.println("\n✓ Connected! IP: " + WiFi.localIP().toString());
    MDNS.begin("esp32-receiver");
  }
  else
    Serial.println("\n✗ Connection Failed");
}

void connectMQTT()
{
  if (WiFi.status() != WL_CONNECTED)
    return;

  // Tự động tìm IP Broker
  if (discovered_mqtt_broker == "")
  {
    Serial.println("mDNS: Đang tìm MQTT Broker trong mạng...");
    int n = MDNS.queryService("mqtt", "tcp");
    if (n > 0)
    {
      discovered_mqtt_broker = MDNS.address(0).toString();
      Serial.print("mDNS: Đã tìm thấy Broker tại: ");
      Serial.println(discovered_mqtt_broker);
      mqtt_client.setServer(discovered_mqtt_broker.c_str(), mqtt_port);
    }
    else
    {
      // Fallback: dùng IP được cấu hình nếu mDNS không tìm thấy
      if (strlen(mqtt_broker_fallback_ip) > 0)
      {
        discovered_mqtt_broker = String(mqtt_broker_fallback_ip);
        Serial.print("mDNS: Không tìm thấy Broker. Dùng Fallback IP: ");
        Serial.println(discovered_mqtt_broker);
        mqtt_client.setServer(discovered_mqtt_broker.c_str(), mqtt_port);
      }
      else
      {
        Serial.println("mDNS: Không tìm thấy Broker nào trong mạng.");
        delay(2000);
        return; // Thoát ra để loop() gọi lại connectMQTT và quét lại
      }
    }
  }

  while (!mqtt_client.connected())
  {
    Serial.print("→ Connecting MQTT: ");
    Serial.println(discovered_mqtt_broker);
    if (mqtt_client.connect(mqtt_client_id))
    {
      Serial.println("✓ MQTT Connected!");
      Serial.print("✓ Subscribed to topic: ");
      Serial.println(mqtt_topic_gesture);
      mqtt_client.subscribe(mqtt_topic_gesture);
    }
    else
    {
      Serial.print("✗ MQTT Failed, rc=");
      Serial.println(mqtt_client.state());
      Serial.println("  (Sẽ thử lại sau 2 giây...)");
      delay(2000);

      // Nếu lỗi, reset để tìm lại mDNS ở lần sau
      discovered_mqtt_broker = "";
      return; // Thoát để loop() gọi lại connectMQTT và quét lại
    }
  }
}

void parse_gesture_message(const char *json_string)
{
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, json_string))
    return;

  const char *gesture = doc["gesture"] | "UNKNOWN";
  Serial.print("  Gesture: ");
  Serial.println(gesture);

  if (strstr(gesture, "Start"))
  {
    buzzerStartBeep();
    return;
  }

  // LED Control
  if (strstr(gesture, "Light1On"))
  {
    Serial.print("  → Lệnh: Light1On | State hiện tại: ");
    Serial.println(light1_state ? "ON" : "OFF");
    if (!light1_state)
    {
      light1_state = true;
      buzzerActionBeep(); // Beep trước khi thực hiện
      led1On();
    }
  }
  else if (strstr(gesture, "Light1Off"))
  {
    if (light1_state)
    {
      light1_state = false;
      buzzerActionBeep();
      led1Off();
    }
  }
  else if (strstr(gesture, "Light2On"))
  {
    Serial.print("  → Lệnh: Light2On | State hiện tại: ");
    Serial.println(light2_state ? "ON" : "OFF");
    if (!light2_state)
    {
      light2_state = true;
      buzzerActionBeep();
      led2On();
    }
  }
  else if (strstr(gesture, "Light2Off"))
  {
    if (light2_state)
    {
      light2_state = false;
      buzzerActionBeep();
      led2Off();
    }
  }

  // Fan Speed Control
  else if (strstr(gesture, "FanOff"))
  {
    if (fan_speed_state != 0)
    {
      fan_speed_state = 0;
      buzzerActionBeep();
      rampFanTo(0);
    }
  }
  else if (strstr(gesture, "FanSpeed1"))
  {
    if (fan_speed_state != 1)
    {
      fan_speed_state = 1;
      buzzerActionBeep();
      rampFanTo(30);
    }
  }
  else if (strstr(gesture, "FanSpeed2"))
  {
    if (fan_speed_state != 2)
    {
      fan_speed_state = 2;
      buzzerActionBeep();
      rampFanTo(65);
    }
  }
  else if (strstr(gesture, "FanSpeed3"))
  {
    if (fan_speed_state != 3)
    {
      fan_speed_state = 3;
      buzzerActionBeep();
      rampFanTo(100);
    }
  }

  // Servo Control (Fan Direction)
  else if (strstr(gesture, "FanLeft"))
  {
    if (servo1AngleIndex > 0)
    {
      servo1AngleIndex--;
      buzzerActionBeep();
      moveServo1Smooth(servo1Angles[servo1AngleIndex]);
    }
  }
  else if (strstr(gesture, "FanRight"))
  {
    if (servo1AngleIndex < servo1MaxIndex)
    {
      servo1AngleIndex++;
      buzzerActionBeep();
      moveServo1Smooth(servo1Angles[servo1AngleIndex]);
    }
  }
  else if (strstr(gesture, "FanDown"))
  {
    if (servo2AngleIndex < servo2MaxIndex)
    {
      servo2AngleIndex++;
      buzzerActionBeep();
      moveServo2Smooth(servo2Angles[servo2AngleIndex]);
    }
  }
  else if (strstr(gesture, "FanUp"))
  {
    if (servo2AngleIndex > 0)
    {
      servo2AngleIndex--;
      buzzerActionBeep();
      moveServo2Smooth(servo2Angles[servo2AngleIndex]);
    }
  }
}

void mqtt_callback(char *topic, byte *payload, unsigned int length)
{
  Serial.print("  [MQTT] Nhận được message từ topic: ");
  Serial.println(topic);
  Serial.print("  [MQTT] Độ dài payload: ");
  Serial.println(length);

  char message[length + 1];
  memcpy(message, payload, length);
  message[length] = '\0';

  Serial.print("  [MQTT] Payload: ");
  Serial.println(message);

  parse_gesture_message(message);
}

// SETUP & LOOP
void setup()
{
  Serial.begin(115200);
  pinMode(LED_IN1, OUTPUT);
  pinMode(LED_IN2, OUTPUT);
  pinMode(LED_IN3, OUTPUT);
  pinMode(LED_IN4, OUTPUT);
  pinMode(FAN_IN1, OUTPUT);
  pinMode(FAN_IN2, OUTPUT);
  pinMode(FAN_ENA, OUTPUT);
  digitalWrite(FAN_IN1, HIGH);
  digitalWrite(FAN_IN2, LOW);

  servo1.attach(servo1Pin, 500, 2400);
  servo2.attach(servo2Pin, 500, 2400);
  servo1.write(90);
  servo2.write(35);

  initBuzzer();
  connectWiFi();
  mqtt_client.setCallback(mqtt_callback);
  Serial.println("SYSTEM READY");
}

void loop()
{
  if (!mqtt_client.connected())
    connectMQTT();
  mqtt_client.loop();
  delay(10);
}
