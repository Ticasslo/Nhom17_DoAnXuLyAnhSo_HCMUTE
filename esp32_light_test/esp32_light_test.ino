// L298N #1 - Điều khiển 2 LED
#define LED_IN1 14
#define LED_IN2 27
#define LED_IN3 26
#define LED_IN4 25

// L298N #2 - Điều khiển Quạt
const int FAN_ENA = 18;
const int FAN_IN1 = 32;
const int FAN_IN2 = 33;

// PWM config cho quạt
const int freq = 5000;
const int resolution = 8;

// Thời gian ramp cho quạt
const unsigned long rampMillis = 800;
const int rampSteps = 25;

// Trạng thái tốc độ quạt hiện tại
static int currentFanPercent = 0;

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("=== ESP32: 2 LED + Fan Control ===");

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

  if (!ledcAttach(FAN_ENA, freq, resolution)) {
    Serial.println("ERROR: ledcAttach failed!");
  }

  setFanSpeed(0);
  Serial.println("Setup complete!");
  delay(500);
}

// Hàm điều khiển LED
void led1On() {
  digitalWrite(LED_IN1, HIGH);
  digitalWrite(LED_IN2, LOW);
  Serial.println("LED 1: ON");
}

void led1Off() {
  digitalWrite(LED_IN1, LOW);
  digitalWrite(LED_IN2, LOW);
  Serial.println("LED 1: OFF");
}

void led2On() {
  digitalWrite(LED_IN3, HIGH);
  digitalWrite(LED_IN4, LOW);
  Serial.println("LED 2: ON");
}

void led2Off() {
  digitalWrite(LED_IN3, LOW);
  digitalWrite(LED_IN4, LOW);
  Serial.println("LED 2: OFF");
}

void allLedOff() {
  led1Off();
  led2Off();
}

// Hàm điều khiển Quạt
void setFanSpeed(int percent) {
  if (percent < 0) percent = 0;
  if (percent > 100) percent = 100;
  int pwmVal = map(percent, 0, 100, 0, 255);
  
  if (!ledcWrite(FAN_ENA, pwmVal)) {
    Serial.println("WARNING: ledcWrite failed");
  }
}

void rampFanTo(int targetPercent) {
  if (targetPercent < 0) targetPercent = 0;
  if (targetPercent > 100) targetPercent = 100;

  int start = currentFanPercent;
  int end = targetPercent;
  
  if (start == end) return;

  // Kick-start: Nếu quạt đang tắt và muốn bật, khởi động ở 100% trước
  if (start == 0 && end > 0) {
    Serial.println("Kick-start: 100% for 500ms");
    setFanSpeed(100);
    delay(500);
    start = 100;
  }

  for (int step = 1; step <= rampSteps; ++step) {
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

// LOOP - Demo điều khiển
void loop() {
  // Demo 1: Bật LED 1, quạt 70%
  Serial.println("\n--- LED1 ON + Fan 70% ---");
  led1On();
  rampFanTo(70);
  delay(3000);

  // Demo 2: Bật LED 2, quạt 85%
  Serial.println("\n--- LED2 ON + Fan 85% ---");
  led1Off();
  led2On();
  rampFanTo(85);
  delay(3000);

  // Demo 3: Cả 2 LED sáng, quạt 100%
  Serial.println("\n--- Both LED ON + Fan 100% ---");
  led1On(); 
  led2On();
  rampFanTo(100);
  delay(3000);

  // Demo 4: Tắt hết
  Serial.println("\n--- All OFF ---");
  allLedOff();
  rampFanTo(0);
  delay(3000);
}