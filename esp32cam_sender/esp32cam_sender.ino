// ESP32-S3 N16R8 + OV2640
// Stream MJPEG qua HTTP: /stream (multipart)
// Board: ESP32-S3, Flash 16MB, PSRAM 8MB. WiFi STA.

#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_http_server.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Debug Mode
// Bỏ comment dòng dưới để bật debug (Serial prints + FPS counter)
// #define DEBUG_MODE

// WiFi
const char *ssid = "ilovehcmute";
const char *password = "910JQKA2";

// Camera quality/FPS trade-off
// HVGA (480x320) - cân bằng chất lượng và FPS
#define FRAME_SIZE FRAMESIZE_HVGA // 480 x 320
#define JPEG_QUALITY 38
// jpeg_quality: số CÀNG LỚN -> chất lượng THẤP hơn -> ảnh NHỎ hơn -> FPS CAO & mượt hơn
#define FB_COUNT 2 // Buffering (ESP32-S3 có 8MB PSRAM)

// Pinout ESP32-S3 N16R8 + OV2640 (ESP32-S3-EYE layout)
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5

#define Y9_GPIO_NUM 16
#define Y8_GPIO_NUM 17
#define Y7_GPIO_NUM 18
#define Y6_GPIO_NUM 12
#define Y5_GPIO_NUM 10
#define Y4_GPIO_NUM 8
#define Y3_GPIO_NUM 9
#define Y2_GPIO_NUM 11
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13

httpd_handle_t stream_httpd = nullptr;

#ifdef DEBUG_MODE
// FPS counter cho stream (chỉ khi DEBUG_MODE)
static uint32_t fps_count = 0;
static uint32_t fps_last_ms = 0;
#endif

static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";
static const char *STREAM_BOUNDARY = "\r\n--frame\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req)
{
  esp_err_t res = ESP_OK;
  httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  static uint32_t last = 0; // Giới hạn FPS để tránh quá tải khi chạy lâu

  while (true)
  {
    // Giới hạn FPS ~20 FPS (50ms/frame) để tránh quá tải khi chạy lâu
    uint32_t now = millis();
    if (now - last < 50) // ~20 FPS (1000ms / 20 ≈ 50ms)
    {
      vTaskDelay(pdMS_TO_TICKS(2));
      continue;
    }
    last = now;

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb)
    {
      res = ESP_FAIL;
      break;
    }

    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK)
    {
      char part_buf[64];
      size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, (unsigned)fb->len);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK)
    {
      // Gửi theo chunks 8KB để giảm overhead
      size_t sent = 0;
      const size_t chunk_size = 8192; // 8KB chunks
      while (sent < fb->len && res == ESP_OK)
      {
        size_t to_send = (fb->len - sent > chunk_size) ? chunk_size : (fb->len - sent);
        res = httpd_resp_send_chunk(req, (const char *)(fb->buf + sent), to_send);
        sent += to_send;
      }
    }

    esp_camera_fb_return(fb);

    // vTaskDelay nhỏ để tránh watchdog timeout và cho task khác chạy
    // Giúp WiFi task và các task khác có cơ hội chạy
    if (res == ESP_OK)
    {
      vTaskDelay(pdMS_TO_TICKS(1)); // 1ms delay
    }

#ifdef DEBUG_MODE
    // FPS counter (chỉ khi DEBUG_MODE)
    fps_count++;
    uint32_t now_debug = millis();
    if (now_debug - fps_last_ms >= 1000)
    {
      Serial.printf("Stream FPS: %u\n", fps_count);
      fps_count = 0;
      fps_last_ms = now_debug;
    }
#endif

    if (res != ESP_OK)
    {
      break;
    }
  }

  return res;
}

void startCameraServer()
{
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32769;       // tránh xung đột cổng điều khiển mặc định
  config.max_uri_handlers = 1;    // Chỉ 1 endpoint: /stream (ít handler → scheduling đều hơn → performance tốt hơn)
  config.max_open_sockets = 1;    // Client (hiện tại tối ưu cho 1 client)
  config.lru_purge_enable = true; // dọn client cũ
  config.backlog_conn = 5;        // hàng đợi kết nối

  httpd_uri_t stream_uri = {
      .uri = "/stream",
      .method = HTTP_GET,
      .handler = stream_handler,
      .user_ctx = nullptr};

  if (httpd_start(&stream_httpd, &config) == ESP_OK)
  {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.printf("HTTP server started on port %u (endpoint: /stream)\n", config.server_port);
  }
  else
  {
    Serial.println("Không thể khởi động HTTP server");
  }
}

// ====== FreeRTOS Tasks ======
void wifiTask(void *pvParameters)
{
  const uint32_t RECONNECT_TIMEOUT_MS = 30000; // 30 giây timeout
  const uint32_t CHECK_INTERVAL_MS = 2000;     // Check mỗi 2 giây khi disconnected
  const uint32_t NORMAL_CHECK_MS = 30000;      // Check mỗi 30 giây khi connected
  const uint8_t MAX_RETRIES = 3;               // Số lần retry trước khi reset WiFi

  uint8_t retry_count = 0;
  bool was_connected = true;

  while (true)
  {
    wl_status_t status = WiFi.status();

    if (status != WL_CONNECTED)
    {
      if (was_connected)
      {
        // Vừa mất kết nối
        Serial.println("WiFi disconnected!");
        retry_count = 0;
        was_connected = false;
      }

      // Thử reconnect với timeout
      uint32_t start_time = millis();
      Serial.print("Attempting to reconnect");

      WiFi.reconnect();

      while (WiFi.status() != WL_CONNECTED && (millis() - start_time) < RECONNECT_TIMEOUT_MS)
      {
        vTaskDelay(pdMS_TO_TICKS(500));
        Serial.print(".");
      }
      Serial.println();

      if (WiFi.status() == WL_CONNECTED)
      {
        // Reconnect thành công
        Serial.println("WiFi reconnected!");
        Serial.print("IP address: ");
        Serial.println(WiFi.localIP());
        retry_count = 0;
        was_connected = true;
        esp_wifi_set_ps(WIFI_PS_NONE); // Tắt power save để giảm lag
      }
      else
      {
        // Reconnect thất bại
        retry_count++;
        Serial.printf("Reconnect failed (attempt %d/%d)\n", retry_count, MAX_RETRIES);

        if (retry_count >= MAX_RETRIES)
        {
          // Reset WiFi sau nhiều lần thất bại
          Serial.println("Too many failures, resetting WiFi...");
          WiFi.disconnect(true);
          vTaskDelay(pdMS_TO_TICKS(1000));
          WiFi.mode(WIFI_STA);
          WiFi.begin(ssid, password);
          retry_count = 0;
        }
      }

      vTaskDelay(pdMS_TO_TICKS(CHECK_INTERVAL_MS));
    }
    else
    {
      // Đã connected
      if (!was_connected)
      {
        // Vừa reconnect thành công
        Serial.println("WiFi connected!");
        Serial.print("IP address: ");
        Serial.println(WiFi.localIP());
        was_connected = true;
      }
      vTaskDelay(pdMS_TO_TICKS(NORMAL_CHECK_MS));
    }
  }
}

void setup()
{
  Serial.begin(115200);
#ifdef DEBUG_MODE
  Serial.setDebugOutput(true);
  Serial.println();
#endif

  // Set CPU frequency
  setCpuFrequencyMhz(160);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 16000000; // 16 MHz (giảm nhiệt camera)
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAME_SIZE;
  config.jpeg_quality = JPEG_QUALITY;
  config.fb_count = FB_COUNT;
  config.grab_mode = CAMERA_GRAB_LATEST;   // luôn lấy frame mới nhất
  config.fb_location = CAMERA_FB_IN_PSRAM; // dùng PSRAM

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK)
  {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    Serial.println("Kiểm tra pinout và kết nối camera.");
    vTaskDelay(pdMS_TO_TICKS(3000)); // Đợi 3 giây để Serial log kịp gửi
    ESP.restart();                   // Restart để thử lại
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s != nullptr)
  {
    // Tuning cơ bản để giảm lag / hình ổn định
    s->set_framesize(s, FRAME_SIZE);
    s->set_vflip(s, 0);
    s->set_hmirror(s, 0);
    s->set_brightness(s, 0);
    s->set_contrast(s, 1); // tăng contrast cho chuyển động
    s->set_saturation(s, 0);
    s->set_gainceiling(s, GAINCEILING_2X);
    s->set_whitebal(s, 1); // AWB on
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);
    s->set_ae_level(s, 0);
    s->set_aec_value(s, 300);
    s->set_gain_ctrl(s, 1);
    s->set_agc_gain(s, 0);
    s->set_bpc(s, 0);       // tắt black pixel correction (giảm nhiệt)
    s->set_wpc(s, 0);       // tắt white pixel correction (giảm nhiệt)
    s->set_raw_gma(s, 0);   // tắt gamma (giảm nhiệt)
    s->set_lenc(s, 0);      // tắt lens correction (giảm nhiệt)
    s->set_denoise(s, 0);   // tắt denoise để giảm tải
    s->set_sharpness(s, 1); // tăng sharpness nhẹ cho chuyển động
    s->set_special_effect(s, 0);
    s->set_colorbar(s, 0); // tắt color bar
  }

  WiFi.mode(WIFI_STA);
  WiFi.setHostname("ESP32-CAM"); // Đặt hostname để dễ nhận diện
  // WiFi Power: Mặc định có thể là 19.5dBm, 20dBm, hoặc 20.5dBm tùy chuẩn (802.11b/g)
  // Arduino framework thường mặc định ~19.5-20dBm
  // WIFI_POWER_19_5dBm: ~80-100mA | WIFI_POWER_11dBm: ~50-70mA | WIFI_POWER_8_5dBm: ~40-60mA
  WiFi.setTxPower(WIFI_POWER_15dBm);
  WiFi.begin(ssid, password);
  Serial.printf("Connecting to %s", ssid);
  while (WiFi.status() != WL_CONNECTED)
  {
    vTaskDelay(pdMS_TO_TICKS(500));
    Serial.print(".");
  }
  Serial.println();

  // WiFi Power Save: WIFI_PS_NONE (tắt)
  esp_wifi_set_ps(WIFI_PS_NONE); // TẮT power save

  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Khởi tạo mDNS
  if (!MDNS.begin("esp32-cam"))
  {
    Serial.println("Error setting up MDNS responder!");
  }
  else
  {
    Serial.println("mDNS responder started: http://esp32-cam.local");
  }

  startCameraServer();
  Serial.println("Camera stream ready.");
  Serial.println("Mở URL: http://esp32-cam.local/stream");

  // Tạo FreeRTOS task cho WiFi management
  xTaskCreate(
      wifiTask,   // Task function
      "WiFiTask", // Task name
      4096,       // Stack size
      NULL,       // Parameters
      1,          // Priority
      NULL        // Task handle
  );
}

void loop()
{
  // HTTP server chạy trong background, không cần code trong loop()
  vTaskDelay(pdMS_TO_TICKS(1000));
}
