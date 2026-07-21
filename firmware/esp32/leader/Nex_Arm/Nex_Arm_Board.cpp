#include "Nex_Arm_Board.h"
#include "U8g2lib.h"
#include "Wire.h"

static const char* TAG = "board";

/* 宏函数 获得A的低八位 */
#define GET_LOW_BYTE(A)         ((uint8_t)(A))
/* 宏函数 获得A的高八位 */
#define GET_HIGH_BYTE(A)        ((uint8_t)((A) >> 8))
/* 宏函数 将高低八位合成为十六位 */
#define BYTE_TO_HW(A, B)        ((((uint16_t)(A)) << 8) | (uint8_t)(B))

#define BOOT_BUTTON_PIN         0
#define USER_BUTTON_PIN         2
#define BUZZER_PIN              23
#define I2C_SDA_PIN             26
#define I2C_SCL_PIN             27

#define ADC_PIN                 ADC1_CHANNEL_6    // ADC引脚
#define ADC_WIDTH               ADC_WIDTH_12Bit   // ADC 12位宽度
#define ADC_ATTEN               ADC_ATTEN_DB_2_5   // 6dB衰减器
#define DEFAULT_VREF            1100              // 默认1.1V的参考电压

#define OLED_I2C_ADDR           0x3C

PinButton Boot_Button(BOOT_BUTTON_PIN);
PinButton User_Button(USER_BUTTON_PIN);
U8G2_SSD1306_128X64_NONAME_2_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE, I2C_SCL_PIN, I2C_SDA_PIN);

bool OLED_t::begin()
{
    Wire.setPins(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.begin();
    Wire.beginTransmission(OLED_I2C_ADDR);
    if(!Wire.endTransmission()) {
        u8g2.begin();
        return true;
    }
    ESP_LOGW(TAG, "OLED initalize failed\n");
    return false;
}

void Bat_t::begin()
{
    int raw;
    int samples_voltage;
    esp_adc_cal_value_t val_type;

    adc1_config_channel_atten(ADC_PIN, ADC_ATTEN);
    val_type = esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN, ADC_WIDTH, DEFAULT_VREF, &adc_chars);
    delay(100);
    raw = adc1_get_raw(ADC_PIN);
    samples_voltage = esp_adc_cal_raw_to_voltage(raw, &adc_chars);
    samples_voltage = (int)(((r1 + r2) / r2) * (float)samples_voltage);
    voltage = samples_voltage;
}

void Bat_t::update()
{
    int raw;
    int samples_voltage;
    int sum = 0;

    raw = adc1_get_raw(ADC_PIN);
    samples_voltage = esp_adc_cal_raw_to_voltage(raw, &adc_chars);
    samples_voltage = (int)(((r1 + r2) / r2) * (float)samples_voltage);
    filter_buf[filter_index] = samples_voltage;
    filter_index = (filter_index + 1) % WINDOWS_SIZE;

    if(filter_buf[WINDOWS_SIZE - 1] != 0) {
        for (uint8_t i = 0; i < WINDOWS_SIZE; i++) {
            sum += filter_buf[i];
        }    

        voltage = sum / WINDOWS_SIZE;
    }
 
    // ESP_LOGI(TAG, "voltage: %d\n", voltage);
}

int Bat_t::get_voltage()
{
    return voltage;
}

void Button_t::update()
{
    Boot_Button.update();
    User_Button.update();
}

bool Button_t::is_clicked(uint8_t id)
{
    switch (id)
    {
    case 0:
        return Boot_Button.isClick();

    case 1:
        return User_Button.isClick();

    default:
        return false;
    }
}

void Buzzer_t::update()
{
    switch(stage) {
    case BUZZER_STAGE_START_NEW_CYCLE:
        if(ticks_on > 0) {
            tone(BUZZER_PIN, freq);
            if(ticks_off > 0) {  
                ticks_count = 0;
                stage = BUZZER_STAGE_WATTING_OFF; 
            }
            else {
                stage = BUZZER_STAGE_IDLE; 
            }
        }
        else { 
            noTone(BUZZER_PIN);
            stage = BUZZER_STAGE_IDLE; 
        }
        break;
        
    case BUZZER_STAGE_WATTING_OFF:
        ticks_count += update_period;
        if(ticks_count >= ticks_on) { 
            noTone(BUZZER_PIN);
            stage = BUZZER_STAGE_WATTING_PERIOD_END;
        }
        break;

    case BUZZER_STAGE_WATTING_PERIOD_END: 
        ticks_count += update_period;
        if(ticks_count >= (ticks_off + ticks_on)) {
            ticks_count -= (ticks_off + ticks_on);
            if(times == 1) { 
                noTone(BUZZER_PIN);
                stage = BUZZER_STAGE_IDLE;  
            }
            else {
                tone(BUZZER_PIN, freq);
                times = times == 0 ? 0 : times - 1;
                stage = BUZZER_STAGE_WATTING_OFF;
            }
        }
        break;

    case BUZZER_STAGE_IDLE:
        break;

    default:
        break;
    }
}    

void Buzzer_t::begin()
{
    allow_change = true;
    this->stage = BUZZER_STAGE_START_NEW_CYCLE;
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);
    ledcAttachPin(BUZZER_PIN, 0);

}

bool Buzzer_t::on(uint16_t freq)
{
  if(allow_change == true) {
    this->ticks_on = 1;
    this->ticks_off = 0;
    this->times = 0;
    this->freq = freq;
    this->stage = BUZZER_STAGE_START_NEW_CYCLE;
    return true;
  }
  return false;
}

bool Buzzer_t::off(void)
{
  if(allow_change == true) {
    this->ticks_on = 0;
    this->ticks_off = 1;
    this->times = 0;
    this->stage = BUZZER_STAGE_START_NEW_CYCLE;
    return true;
  }
  return false;
}

bool Buzzer_t::set(uint32_t on_time , uint32_t off_time , uint16_t times, uint16_t freq)
{
  if(allow_change == true) {
    this->ticks_on = on_time;
    this->ticks_off = off_time;
    this->times = times;
    this->freq = freq;
    this->stage = BUZZER_STAGE_START_NEW_CYCLE;
    return true;
  }
  return false;
}

void HW_Board::begin()
{
  if(!SPIFFS.begin(true)) {
    ESP_LOGI(TAG, "SPIFFS Mount Failed\n");
    return;
  }

  bat.begin();
  buzzer.begin();
  oled.begin();
}

void HW_Board::list_action_group_dir()
{
  ESP_LOGI("Robot", "Listing directory: /\n");
  File root = SPIFFS.open("/");
  if(!root) {
    ESP_LOGI("Robot", "- failed to open directory\n");
    return;
  }
  if(!root.isDirectory()){
    ESP_LOGI("Robot", " - not a directory\n");
    return;
  }

  File file = root.openNextFile();
  while(file) {
    if(file.isDirectory()) {
      ESP_LOGI("Robot", "DIR:%s\n", file.name());
    } 
    else {
      ESP_LOGI("Robot", "FILE:%s SIZE: %d\n", file.name(), file.size());
    }
    file = root.openNextFile();
  }
}

void HW_Board::action_group_run(uint8_t id)
{
  uint8_t offset;
  uint8_t frame_index;
  uint8_t control_num;
  uint8_t count = 0;
  uint8_t buf[58] = {0}; /* frame_index|control_num|time_l|time_h|id|duty_l|duty_h|... */
  uint16_t move_time;
  // ServoArg_t servos[18];

  // func_state = ACTION_GROUP;
  act_state = READ_FRAME_NUM;
  File file = SPIFFS.open("/ActionGroup" + String(id) + ".rob", FILE_READ);
  if(!file) {
    ESP_LOGI("Robot", "Failed to open file for reading\n");
  }  

  while(file.available()) {
    switch(act_state) {
      case READ_FRAME_NUM: /*读取帧头*/
        act_read_frame_num = (uint8_t)file.read();
        act_state = READ_FRAME_DATA;
        break;

      case READ_FRAME_DATA:
        file.read(buf, sizeof(buf));
        control_num = buf[1];
        frame_index = buf[0];
        move_time = BYTE_TO_HW(buf[3], buf[2]); 
        if(id == 0) {
          Serial.printf("$$>%d<$$", frame_index);
        }
        ESP_LOGI("Robot", "id: %d frame_num: %d frame_index: %d control_num: %d move_time: %d\n", id, act_read_frame_num, frame_index, control_num, move_time);
        for(uint8_t i = 0; i < control_num; i++) {
          // servos[i].id = buf[4 + i * 3];
          // servos[i].duty = BYTE_TO_HW(buf[6 + i * 3], buf[5 + i * 3]); 
          // ESP_LOGI("Robot", "id: %d duty: %d\n",  servos[i].id, servos[i].duty);
        }
        // servo.multi_set(servos, control_num, move_time);
        delay(move_time);
        if(frame_index == act_read_frame_num) {
          act_state = ACT_STOP;
          if(id == 0) {
            Serial.printf("$$>end<$$");
          }
        }
        break;

      default:
        break;
    }  

    if(act_state == ACT_STOP) {
      break;
    }
  }

  file.close();
}

bool HW_Board::action_group_download(uint8_t id, uint8_t *data, size_t length)
{
  size_t written;
  size_t len;
  uint8_t frame_index = data[2];

  if(id > 50) {
    ESP_LOGI("Robot", "id error!\n");
    return false;
  }

  if(frame_index > 254) {
    ESP_LOGI("Robot", "Frame index error!\n");
    return false;    
  }

  if(frame_index == 1) {
    File file = SPIFFS.open("/ActionGroup" + String(id) + ".rob", FILE_WRITE);
    if(!file) {
      ESP_LOGI("Robot", "- failed to open file for downloading\n");
      return false;
    }
    len = length - 1;
    written = file.write(&data[1], len);
    file.close();
  }
  else {
    File file = SPIFFS.open("/ActionGroup" + String(id) + ".rob", FILE_APPEND);
    if(!file) {
      ESP_LOGI("Robot", "- failed to open file for downloading\n");
      return false;
    }
    len = length - 2;
    written = file.write(&data[2], len);
    file.close();
  }

  if(written == len) {
    ESP_LOGI("Robot", "- %u bytes downloading\n", written);
  }
  else {
    ESP_LOGI("Robot", "- downloading failed, only wrote %u of %u bytes\n", written, length);
  }
  return true;
}

bool HW_Board::action_group_erase(uint8_t id)
{
    if(id > 50) {
        ESP_LOGI("Robot", "ID ERROR!\n");
        return false;
    }

    File file = SPIFFS.open("/ActionGroup" + String(id) + ".rob", FILE_WRITE);
    if(!file) {
        ESP_LOGI("Robot", "- failed to open file for writing\n");
        return false;
    }
    file.close();
    return true;
}

void HW_Board::action_group_stop(void)
{
    act_state = ACT_STOP;
} 