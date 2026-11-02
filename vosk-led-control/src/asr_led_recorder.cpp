#include <Arduino.h>
#include <driver/i2s.h>

// INMP441 wiring: SD -> GPIO 13, SCK -> GPIO 14, WS -> GPIO 15, L/R -> GND.
constexpr i2s_port_t I2S_PORT = I2S_NUM_0;
constexpr int MIC_DATA_PIN = 13;
constexpr int MIC_CLOCK_PIN = 14;
constexpr int MIC_WORD_SELECT_PIN = 15;
constexpr uint8_t ONBOARD_RGB_PIN = 48;

bool microphoneReady = false;

void setLedFromCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command == "LED RED") {
    neopixelWrite(ONBOARD_RGB_PIN, 40, 0, 0);
  } else if (command == "LED GREEN") {
    neopixelWrite(ONBOARD_RGB_PIN, 0, 40, 0);
  } else if (command == "LED BLUE") {
    neopixelWrite(ONBOARD_RGB_PIN, 0, 0, 40);
  } else if (command == "LED WHITE") {
    neopixelWrite(ONBOARD_RGB_PIN, 30, 30, 30);
  } else if (command == "LED YELLOW") {
    neopixelWrite(ONBOARD_RGB_PIN, 40, 30, 0);
  } else if (command == "LED PURPLE") {
    neopixelWrite(ONBOARD_RGB_PIN, 30, 0, 40);
  } else if (command == "LED CYAN") {
    neopixelWrite(ONBOARD_RGB_PIN, 0, 35, 35);
  } else if (command == "LED ORANGE") {
    neopixelWrite(ONBOARD_RGB_PIN, 40, 10, 0);
  } else if (command == "LED EMERGENCY") {
    neopixelWrite(ONBOARD_RGB_PIN, 50, 0, 0);
    delay(250);
    neopixelWrite(ONBOARD_RGB_PIN, 0, 0, 50);
    delay(250);
    neopixelWrite(ONBOARD_RGB_PIN, 50, 0, 0);
    delay(250);
    neopixelWrite(ONBOARD_RGB_PIN, 0, 0, 50);
    delay(250);
    neopixelWrite(ONBOARD_RGB_PIN, 0, 0, 0);
  } else if (command == "LED OFF") {
    neopixelWrite(ONBOARD_RGB_PIN, 0, 0, 0);
  }
}

void setupMicrophone() {
  i2s_config_t config = {};
  config.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
  config.sample_rate = 16000;
  config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  config.dma_buf_count = 8;
  config.dma_buf_len = 256;

  i2s_pin_config_t pins = {};
  pins.bck_io_num = MIC_CLOCK_PIN;
  pins.ws_io_num = MIC_WORD_SELECT_PIN;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = MIC_DATA_PIN;

  microphoneReady = i2s_driver_install(I2S_PORT, &config, 0, nullptr) == ESP_OK &&
                    i2s_set_pin(I2S_PORT, &pins) == ESP_OK;
  i2s_zero_dma_buffer(I2S_PORT);
}

void streamAudio() {
  int32_t raw[256];
  int16_t pcm[256];
  Serial.println("ASR_READY 16000 PCM16");
  neopixelWrite(ONBOARD_RGB_PIN, 0, 0, 30);

  while (true) {
    if (Serial.available()) {
      String command = Serial.readStringUntil('\n');
      command.trim();
      if (command.startsWith("STOP")) break;
      setLedFromCommand(command);
    }

    size_t bytesRead = 0;
    if (i2s_read(I2S_PORT, raw, sizeof(raw), &bytesRead, pdMS_TO_TICKS(100)) != ESP_OK) {
      continue;
    }
    const size_t sampleCount = bytesRead / sizeof(raw[0]);
    for (size_t i = 0; i < sampleCount; ++i) {
      pcm[i] = static_cast<int16_t>(raw[i] >> 16);
    }
    Serial.write(reinterpret_cast<const uint8_t *>(pcm), sampleCount * sizeof(pcm[0]));
  }

  Serial.flush();
  neopixelWrite(ONBOARD_RGB_PIN, 0, 0, 0);
  Serial.println("ASR_STOPPED");
}

void setup() {
  Serial.begin(460800);
  Serial.setTimeout(10);
  neopixelWrite(ONBOARD_RGB_PIN, 0, 0, 0);
  setupMicrophone();
  Serial.printf("ASR_RECORDER %s\n", microphoneReady ? "READY" : "I2S_ERROR");
}

void loop() {
  if (!Serial.available()) {
    delay(10);
    return;
  }

  const String command = Serial.readStringUntil('\n');
  if (command.startsWith("START") && microphoneReady) {
    streamAudio();
  } else if (command.startsWith("START")) {
    Serial.println("ERROR I2S initialization");
  } else {
    setLedFromCommand(command);
  }
}
