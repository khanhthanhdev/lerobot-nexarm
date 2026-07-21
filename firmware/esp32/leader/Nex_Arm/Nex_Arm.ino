#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "HX_30HM.h"
#include "CommProtocol.h"

#define WIFI_CHANNEL        2
#define BUZZER_PIN          23
#define SEND_INTERVAL_MS    4

#define JOYSTICK_SW_PIN     4
#define JOYSTICK_X_PIN      39
#define JOYSTICK_Y_PIN      36

typedef struct __attribute__((packed)) {
    uint32_t seq;
    int16_t pos[6]; 
} ArmPacket_t;

ArmPacket_t txPacket;
uint8_t broadcastMac[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
uint8_t idList[6] = {1, 2, 3, 4, 5, 6}; 
unsigned long lastSendTime = 0;
unsigned long lastDebugTime = 0; 

bool isJoystickMode = false;
float current_x = 200.0;
float current_y = 0.0;
float current_z = 200.0;
float current_pitch = 0;
float current_roll = 0;
int lastBtnState = HIGH;
unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 200;

int joy_center_x = 2048;
int joy_center_y = 2048;

int16_t last_valid_pos[6] = {2048, 2048, 2048, 2048, 2048, 2048};

// ---- LeRobot 模式 ----
static bool lerobotMode = false;
CommProtocol_t usb_protocol;

static void send_reply(uint8_t cmd, uint8_t *data, uint8_t len) {
    uint8_t tx_len = usb_protocol.tx_packet_complete(0xFF, cmd, data, len);
    Serial.write((const uint8_t *)&usb_protocol.tx_packet, tx_len);
}

static void handle_lerobot_cmd(PacketTypeDef *pkt) {
    uint8_t cmd = pkt->elements.cmd;
    uint8_t *args = pkt->elements.args;

    switch (cmd) {
        case 68: { // CMD_LEROBOT_MODE — 进入/退出 LeRobot 模式
            lerobotMode = (args[0] != 0);
            // 不改力矩状态，ESPNow 示教继续正常工作
            uint8_t resp = lerobotMode ? 1 : 0;
            send_reply(68, &resp, 1);
        } break;

        case 96: { // CMD_LR_READ_POS — 逐个 ID 读，避免异常错位
            uint8_t buf[12];
            for (int i = 0; i < 6; i++) {
                int16_t pos = 0;
                while(Serial1.available()) Serial1.read();
                ServoStatus_t status = servo.read_pos(i + 1, &pos);
                if (status.error_bits.bit_rx != 0 || status.error_bits.bit_tx != 0 || pos == 0) {
                    pos = last_valid_pos[i];
                } else {
                    last_valid_pos[i] = pos;
                }
                buf[i * 2]     = (uint8_t)(pos & 0xFF);
                buf[i * 2 + 1] = (uint8_t)((pos >> 8) & 0xFF);
                delayMicroseconds(800);
            }
            send_reply(96, buf, 12);
        } break;

        case 97: { // CMD_LR_WRITE_POS
            int16_t send_data[6][4];
            for (int i = 0; i < 6; i++) {
                int16_t pos = (int16_t)(args[i * 2] | (args[i * 2 + 1] << 8));
                send_data[i][0] = i + 1;
                send_data[i][1] = 0;
                send_data[i][2] = 0;
                send_data[i][3] = pos;
            }
            servo.sync_write_pos_ex(send_data, 6);
        } break;

        case 98: { // CMD_LR_TORQUE
            uint8_t enable = args[0];
            for (uint8_t i = 1; i <= 6; i++) {
                if (enable) servo.enable_torque(i);
                else servo.disable_torque(i);
                delay(2);
            }
            send_reply(98, &enable, 1);
        } break;
    }
}

// ---- 原有功能 ----

void beep(int freq, int duration) {
    ledcSetup(0, freq, 8);
    ledcAttachPin(BUZZER_PIN, 0);
    ledcWrite(0, 128);
    delay(duration);
    ledcWrite(0, 0);
}

void setup() {
    Serial.begin(1000000);
    setCpuFrequencyMhz(240);

    pinMode(JOYSTICK_SW_PIN, INPUT_PULLUP);
    pinMode(JOYSTICK_X_PIN, INPUT);
    pinMode(JOYSTICK_Y_PIN, INPUT);

    servo.begin(Serial1, 1000000, 17, 16); 
    
    // ESPNow 初始化
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    if (esp_now_init() != ESP_OK) {
        return;
    }

    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, broadcastMac, 6);
    peerInfo.channel = WIFI_CHANNEL;  
    peerInfo.encrypt = false;
    esp_now_add_peer(&peerInfo);

    // LeRobot USB 协议解析器
    usb_protocol.begin();
    usb_protocol.register_success_callback(handle_lerobot_cmd);

    // 关闭力矩
    for(int i=1; i<=6; i++) {
        servo.disable_torque(i);
        delay(5);
    }

    // 等待舵机总线稳定后，预读一次真实位置
    delay(300);
    for(int retry = 0; retry < 3; retry++) {
        for(int i = 0; i < 6; i++) {
            int16_t pos = 0;
            while(Serial1.available()) Serial1.read();
            ServoStatus_t status = servo.read_pos(i + 1, &pos);
            if (status.error_bits.bit_rx == 0 && status.error_bits.bit_tx == 0 && pos != 0) {
                last_valid_pos[i] = pos;
            }
            delay(2);
        }
    }

    beep(2000, 200);
    lastSendTime = millis();
}

void loop() {
    unsigned long currentMillis = millis();

    // 始终解析 USB Serial（检测 LeRobot 命令）
    while (Serial.available()) {
        uint8_t buf[64];
        int n = Serial.readBytes(buf, min((int)Serial.available(), (int)64));
        usb_protocol.parsing(buf, n);
    }

    // LeRobot 模式下也继续跑 ESPNow（录制时 ESPNow 控制从臂，PC 旁路录制）
    // 不需要 return

    // ======== 以下是原有 ESPNow 模式，完全不动 ========

    int reading = digitalRead(JOYSTICK_SW_PIN);
    if (reading == LOW && lastBtnState == HIGH && (currentMillis - lastDebounceTime > debounceDelay)) {
        isJoystickMode = !isJoystickMode;
        lastDebounceTime = currentMillis;
        if (isJoystickMode) {
            long sumX = 0;
            long sumY = 0;
            for(int i=0; i<20; i++) {
                sumX += analogRead(JOYSTICK_X_PIN);
                sumY += analogRead(JOYSTICK_Y_PIN);
                delay(2);
            }
            joy_center_x = sumX / 20;
            joy_center_y = sumY / 20;
            beep(1000, 100);
        } else {
            beep(1000, 100);
            delay(100);
            beep(1000, 100);
        }
    }
    lastBtnState = reading;

    if (currentMillis - lastSendTime >= SEND_INTERVAL_MS) {
        lastSendTime = currentMillis;

        if (isJoystickMode) {
            int joyX = analogRead(JOYSTICK_X_PIN);
            int joyY = analogRead(JOYSTICK_Y_PIN);
            
            if (abs(joyY - joy_center_y) < 100) {
                joyY = joy_center_y;
            }
            if (joyY <= joy_center_y) {
                current_x = map(joyY, 0, joy_center_y, 400, 100);
            } else {
                current_x = map(joyY, joy_center_y, 4095, 200, 400);
            }

            if (abs(joyX - joy_center_x) < 20) {
                joyX = joy_center_x;
            }
            if (joyX <= joy_center_x) {
                current_z = map(joyX, 0, joy_center_x, 400, 20);
            } else {
                current_z = map(joyX, joy_center_x, 4095, 200, 400);
            }

            current_y = 0.0;

            uint8_t packet[20];
            packet[0] = 0xFF; packet[1] = 0xFF; packet[2] = 0xFF;
            packet[3] = 0x10; 
            packet[4] = 0x08; 
            
            int16_t x = (int16_t)current_x;
            int16_t y = (int16_t)current_y;
            int16_t z = (int16_t)current_z;
            int16_t pitch = (int16_t)(current_pitch * 10); 
            int16_t roll = (int16_t)(current_roll * 10);   
            int16_t claw = 0;  
            uint16_t time = 0;

            packet[5]  = pitch & 0xFF; packet[6]  = (pitch >> 8) & 0xFF;
            packet[7]  = x & 0xFF;     packet[8]  = (x >> 8) & 0xFF;
            packet[9]  = y & 0xFF;     packet[10] = (y >> 8) & 0xFF;
            packet[11] = z & 0xFF;     packet[12] = (z >> 8) & 0xFF;
            packet[13] = roll & 0xFF;  packet[14] = (roll >> 8) & 0xFF;
            packet[15] = claw & 0xFF;  packet[16] = (claw >> 8) & 0xFF; 
            packet[17] = time & 0xFF;  packet[18] = (time >> 8) & 0xFF; 
            
            uint8_t sum = 0;
            for(int i=2; i<=18; i++) sum += packet[i];
            packet[19] = ~sum; 

            esp_now_send(broadcastMac, packet, 20);

        } else {
             // 示教模式：逐个 ID 读，异常不影响其他舵机
             ArmPacket_t teachPacket;
             teachPacket.seq = currentMillis;
             
             for(int i = 0; i < 6; i++) {
                 int16_t pos = 0;
                 ServoStatus_t status = servo.read_pos(i + 1, &pos);
                 
                 if (status.error_bits.bit_rx == 0 && status.error_bits.bit_tx == 0 && pos != 0) {
                     last_valid_pos[i] = pos;
                 } else if (currentMillis - lastDebugTime > 1000) {
                     Serial.printf("[WARN] ID%d read fail rx=%d tx=%d pos=%d\n", 
                                   i+1, status.error_bits.bit_rx, status.error_bits.bit_tx, pos);
                 }
                 delayMicroseconds(200);
             }

             if (currentMillis - lastDebugTime > 1000) {
                 Serial.printf("[POS] %d %d %d %d %d %d\n",
                     last_valid_pos[0], last_valid_pos[1], last_valid_pos[2],
                     last_valid_pos[3], last_valid_pos[4], last_valid_pos[5]);
                 lastDebugTime = currentMillis;
             }
             
             for(int i = 0; i < 6; i++) {
                 teachPacket.pos[i] = last_valid_pos[i];
             }

             esp_now_send(broadcastMac, (uint8_t*)&teachPacket, sizeof(ArmPacket_t));
        }
    }
}
