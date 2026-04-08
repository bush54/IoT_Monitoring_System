#include <DHT.h>
#include "driver/i2s_std.h"

// DHT11 - Temperature & Humidity
#define DHTPIN 14
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

//  MQ-5 Gas Sensor
#define MQ5_PIN 34
const int gasThreshold = 190;

//  SW-420 Vibration Sensor
#define VIBRATION_PIN 4

//  Limit Switch
#define LIMIT_SWITCH_PIN 27

//  I2S Microphone (NEW DRIVER)
#define I2S_WS   25
#define I2S_SD   33
#define I2S_SCK  26

const int soundThreshold = 1500;
i2s_chan_handle_t rx_handle = NULL;

//  General timing
unsigned long lastReadTime = 0;
const unsigned long readInterval = 10000;

// I2S Microphone Setup 
bool setupMic() {
  esp_err_t err;

  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
  err = i2s_new_channel(&chan_cfg, NULL, &rx_handle);
  if (err != ESP_OK) {
    Serial.printf("i2s_new_channel failed: %d\n", err);
    return false;
  }

  i2s_std_config_t std_cfg = {};
  std_cfg.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000);
  std_cfg.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);

  std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

  std_cfg.gpio_cfg.mclk = I2S_GPIO_UNUSED;
  std_cfg.gpio_cfg.bclk = (gpio_num_t)I2S_SCK;
  std_cfg.gpio_cfg.ws   = (gpio_num_t)I2S_WS;
  std_cfg.gpio_cfg.dout = I2S_GPIO_UNUSED;
  std_cfg.gpio_cfg.din  = (gpio_num_t)I2S_SD;

  std_cfg.gpio_cfg.invert_flags.mclk_inv = false;
  std_cfg.gpio_cfg.invert_flags.bclk_inv = false;
  std_cfg.gpio_cfg.invert_flags.ws_inv   = false;

  err = i2s_channel_init_std_mode(rx_handle, &std_cfg);
  if (err != ESP_OK) {
    Serial.printf("i2s_channel_init_std_mode failed: %d\n", err);
    return false;
  }

  err = i2s_channel_enable(rx_handle);
  if (err != ESP_OK) {
    Serial.printf("i2s_channel_enable failed: %d\n", err);
    return false;
  }

  return true;
}

// Read microphone level 
int getSoundLevel() {
  if (rx_handle == NULL) return 0;

  const int sampleCount = 256;
  int32_t samples[sampleCount];
  size_t bytesRead = 0;

  esp_err_t err = i2s_channel_read(rx_handle, samples, sizeof(samples), &bytesRead, 100);
  if (err != ESP_OK || bytesRead == 0) {
    return 0;
  }

  int count = bytesRead / sizeof(int32_t);
  if (count <= 0) return 0;

  uint64_t sum = 0;

  for (int i = 0; i < count; i++) {
    int32_t sample = samples[i] >> 14;
    if (sample < 0) sample = -sample;
    sum += sample;
  }

  return sum / count;
}

// SETUP
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("System starting...");

  dht.begin();
  pinMode(VIBRATION_PIN, INPUT);
  pinMode(LIMIT_SWITCH_PIN, INPUT_PULLUP);

  if (setupMic()) {
    Serial.println("Microphone initialized with NEW I2S driver.");
  } else {
    Serial.println("Microphone initialization failed.");
  }

  Serial.println("All sensors initialized.");
}


//print
void loop() {
  if (millis() - lastReadTime >= readInterval) {
    lastReadTime = millis();

    // Serial.println("\n====================================");
    // Serial.println("Reading all sensors...");
    // Serial.println("====================================");

    readLimitSwitch();
    readGasSensor();
    readVibrationSensor();
    readSoundSensor();
    readDHTSensor();

    Serial.println("====================================\n");
  }
}

// Limit Switch
void readLimitSwitch() {
  int switchState = digitalRead(LIMIT_SWITCH_PIN);

  Serial.println("Limit Switch");
  Serial.print("Raw state: ");
  Serial.println(switchState);

  if (switchState == LOW) {
    Serial.println("Switch is PRESSED");
  } else {
    Serial.println("Switch is OPEN");
  }

  Serial.println();
}

// MQ-5 Gas Sensor
void readGasSensor() {
  int gasValue = analogRead(MQ5_PIN);

  Serial.println("MQ-5 Gas Sensor");
  Serial.print("Raw gas value: ");
  Serial.println(gasValue);

  if (gasValue > gasThreshold) {
    Serial.println("Gas detected above threshold");
  } else {
    Serial.println("Gas level below threshold");
  }

  Serial.println();
}

// SW-420 Vibration Sensor
void readVibrationSensor() {
  int vibrationState = digitalRead(VIBRATION_PIN);

  Serial.println("SW-420 Vibration Sensor");
  Serial.print("Raw state: ");
  Serial.println(vibrationState);

  if (vibrationState == HIGH) {
    Serial.println("Vibration detected");
  } else {
    Serial.println("No vibration");
  }

  Serial.println();
}

// I2S Microphone
void readSoundSensor() {
  int soundLevel = getSoundLevel();

  Serial.println("I2S Microphone");
  Serial.print("Sound level: ");
  Serial.println(soundLevel);

  if (soundLevel > soundThreshold) {
    Serial.println("Sound detected above threshold");
  } else {
    Serial.println("Sound below threshold");
  }

  Serial.println();
}

// DHT11
void readDHTSensor() {
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  Serial.println("DHT11 Sensor");

  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Failed to read from DHT11");
    Serial.println();
    return;
  }

  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");

  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" C");

  Serial.println();
}