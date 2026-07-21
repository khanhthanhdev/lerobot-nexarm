#include "system_task_handle.h"
#include "Global.h"
#include "U8g2lib.h"
#include "usb_ctrl.h"

static const char* TAG = "system_task";

HW_Board board;

static bool g_get_pos = true;
static uint8_t g_mode = UART_MODE;

static TimerHandle_t TIMER;
static esp_event_loop_handle_t loop_with_sys_task;

ESP_EVENT_DEFINE_BASE(SYS_STATUS_EVENTS);
ESP_EVENT_DEFINE_BASE(SYS_TIMING_EVENTS);

void sys_timer_post_callback(TimerHandle_t xTimer)
{
    static uint32_t count= 0;

    if(xTimer == nullptr) {
        return;
    }

    if(count % BUZZER_UPDATE_PERIOD == 0) {
        ESP_ERROR_CHECK(esp_event_post_to(loop_with_sys_task, 
                                          SYS_TIMING_EVENTS, 
                                          TIMING_EVENT_BUZZER_UPDATE, 
                                          NULL, 
                                          0, 
                                          portMAX_DELAY));
    }

    if(count % OLED_UPDATE_PERIOD == 0) {
        ESP_ERROR_CHECK(esp_event_post_to(loop_with_sys_task, 
                                          SYS_TIMING_EVENTS, 
                                          TIMING_EVENT_OLED_UPDATE, 
                                          NULL, 
                                          0, 
                                          portMAX_DELAY));
    }

    if(count % SERVO_STATUS_UPDATE_PERIOD == 0) {
        ESP_ERROR_CHECK(esp_event_post_to(loop_with_sys_task, 
                                          SYS_TIMING_EVENTS, 
                                          TIMING_EVENT_SERVO_STATUS_UPDATE, 
                                          NULL, 
                                          0, 
                                          portMAX_DELAY));
    }

    if(count % BAT_UPDATE_PERIOD == 0) {
        ESP_ERROR_CHECK(esp_event_post_to(loop_with_sys_task, 
                                          SYS_TIMING_EVENTS, 
                                          TIMING_EVENT_BAT_UPDATE, 
                                          NULL, 
                                          0, 
                                          portMAX_DELAY));
    }

    count += TIMER_PERIOD;
}

static void sys_timer_sub_handler(void*              handler_args, 
                                  esp_event_base_t   base, 
                                  int32_t            id, 
                                  void*              event_data)
{
    int vol;
    uint8_t read_data[11];
    ServoStatus_t status;

    static bool bat_post_flag = false;

    switch(id) {
        case TIMING_EVENT_BUZZER_UPDATE:
            board.buzzer.update();
            break;

        case TIMING_EVENT_OLED_UPDATE:
            break;

        case TIMING_EVENT_SERVO_STATUS_UPDATE:
            if(g_get_pos == true) {
                // arm.status_update();
            }
            break;

        case TIMING_EVENT_BAT_UPDATE:
            board.bat.update();
            vol = board.bat.get_voltage();
            if((vol < 11800 || vol > 13000) && bat_post_flag == false) {
                bat_post_flag = true;
            }
            else if((vol >= 11800 && vol <= 13000) && bat_post_flag == true) {
                bat_post_flag = false;
            }
            break;

        default:
            break;
    }
}


static void error_sub_handler(void*              handler_args, 
                          esp_event_base_t   base, 
                          int32_t            id, 
                          void*              event_data)
{
    ESP_LOGI(TAG, "selftest event triger!");
}

static void boot_button_sub_handler(void*              handler_args, 
                                    esp_event_base_t   base, 
                                    int32_t            id, 
                                    void*              event_data)
{
    g_get_pos = true;
    ESP_LOGI(TAG, "Boot button clicked");

    if (g_mode == UART_MODE) {
        uint8_t key_id = 0; 
        uint8_t len = serial_port.protocol.tx_packet_complete(0xFF, 0x12, &key_id, 1);
        serial_port.uart->write((const uint8_t*)&serial_port.protocol.tx_packet, len);
    }
}

static void user_button_sub_handler(void*              handler_args, 
                                    esp_event_base_t   base, 
                                    int32_t            id, 
                                    void*              event_data)
{
    ESP_LOGI(TAG, "User button clicked");

    if (g_mode == UART_MODE) {
        uint8_t key_id = 1; 
        uint8_t len = serial_port.protocol.tx_packet_complete(0xFF, 0x12, &key_id, 1);
        serial_port.uart->write((const uint8_t*)&serial_port.protocol.tx_packet, len);
    }
}


void func_ctrl_callback(PacketTypeDef* self)
{
    uint8_t len;
    uint8_t version;
    uint16_t bat_level;
    int16_t set_pos[4] = {0};
    float set_pitch;
    float set_rad[4] = {0};
    float set_coord[3] = {0};
    float cal_coord[3] = {0};
    float cal_rad[4] = {0};

    uint8_t set_coord_byte[6] = {0};
    uint8_t set_pos_byte[8] = {0};
    
    if(self->elements.id != 0xFF) {
        servo.tx_frame_write(self->elements.id,
                             self->elements.cmd,
                             self->elements.args,
                             self->elements.length - 2);
        
        g_get_pos = false;                     
    }
    else {
        g_get_pos = true;
        switch (self->elements.cmd)
        {
            case CMD_FIRMWARE_VERSION_CHECK:
                switch(g_mode) {
                    case UART_MODE:
                        version = FIRMWARE_VERSION;
                        len = serial_port.protocol.tx_packet_complete(0xFF, CMD_FIRMWARE_VERSION_CHECK, &version, 1);
                        serial_port.uart->write((const uint8_t*)&serial_port.protocol.tx_packet, len);
                        break;

                    case WIFI_MODE: 
                        break;

                    default:
                        break;
                }
                break;

            case CMD_CHECK_BAT_LEVEL_CHECK:
                switch(g_mode) {
                    case UART_MODE:
                        bat_level = (uint16_t)board.bat.get_voltage();
                        len = serial_port.protocol.tx_packet_complete(0xFF, CMD_CHECK_BAT_LEVEL_CHECK, (uint8_t*)&bat_level, sizeof(bat_level));
                        serial_port.uart->write((const uint8_t*)&serial_port.protocol.tx_packet, len);
                        break;

                    case WIFI_MODE: 
                        break;   

                    default:
                        break;
                }
                break;
            
            case CMD_ACTION_GROUP_RUN:
                break;
            
            case CMD_ACTION_GROUP_STOP:
                break;
            
            case CMD_ACTION_GROUP_DOWNLOAD:
                break;
                
            case CMD_FKINE_RESULT_GET:
                break;
            
            case CMD_IKINE_RESULT_GET:
                break;
            
            case CMD_COORDINATE_SET:
                break;

            case 0x10:
            {
                if(self->elements.length >= 6) {
                    uint16_t freq = BYTE_TO_HW(self->elements.args[0], self->elements.args[1]);
                    uint16_t time_ms = BYTE_TO_HW(self->elements.args[2], self->elements.args[3]);
                    board.buzzer.set(time_ms / 20, 0, 1, freq);
                }
            }
            break;

            case 0x11:
            {
                if(self->elements.length >= 3) {
                    uint8_t text_len = self->elements.length - 3; 
                    if(text_len > 0) {
                        char text_buff[21]; 
                        if(text_len > 20) text_len = 20; 
                        memcpy(text_buff, &self->elements.args[1], text_len);
                        text_buff[text_len] = '\0'; 
                        ESP_LOGI(TAG, "OLED Text: %s", text_buff);
                    }
                }
            }
            break;

            default:
                break;
        }
    }
}

void system_loop_handler()
{
    serial_port.rec_handler();
    board.button.update();
    
    uint8_t len = servo.uart->available();
        
    if(len > 0) {
        uint8_t buff[len] = {0};
        servo.uart->read(buff, len);
        serial_port.uart->write(buff, len);
    }
        
    if(board.button.is_clicked(0)) {
        ESP_ERROR_CHECK(esp_event_post_to(loop_with_sys_task,
                                            SYS_STATUS_EVENTS,
                                            STATUS_EVENT_BOOT_BUTTON,
                                            NULL,
                                            0,
                                            0));
    }
    
    if(board.button.is_clicked(1)) {
        ESP_ERROR_CHECK(esp_event_post_to(loop_with_sys_task,
                                            SYS_STATUS_EVENTS,
                                            STATUS_EVENT_USER_BUTTON,
                                            NULL,
                                            0,
                                            0));
    }
}

void register_system_task(esp_event_loop_handle_t *event_loop)
{
    loop_with_sys_task = *event_loop;   

    serial_port.begin(Serial, 1000000);
    serial_port.register_ops_callback(func_ctrl_callback);
    servo.begin(Serial1, 1000000);
    board.begin();

    ESP_ERROR_CHECK(esp_event_handler_instance_register_with(loop_with_sys_task,
                                                             SYS_TIMING_EVENTS, 
                                                             ESP_EVENT_ANY_ID, 
                                                             sys_timer_sub_handler, 
                                                             NULL,
                                                             NULL));
                                                        
    ESP_ERROR_CHECK(esp_event_handler_instance_register_with(loop_with_sys_task, 
                                                             SYS_STATUS_EVENTS, 
                                                             STATUS_EVENT_ERROR, 
                                                             error_sub_handler, 
                                                             NULL,
                                                             NULL));

    ESP_ERROR_CHECK(esp_event_handler_instance_register_with(loop_with_sys_task, 
                                                             SYS_STATUS_EVENTS, 
                                                             STATUS_EVENT_BOOT_BUTTON, 
                                                             boot_button_sub_handler, 
                                                             NULL,
                                                             NULL));

    ESP_ERROR_CHECK(esp_event_handler_instance_register_with(loop_with_sys_task, 
                                                             SYS_STATUS_EVENTS, 
                                                             STATUS_EVENT_USER_BUTTON, 
                                                             user_button_sub_handler, 
                                                             NULL,
                                                             NULL));

    TIMER = xTimerCreate("sys_timing_event", 
                         pdMS_TO_TICKS(TIMER_PERIOD), 
                         pdTRUE, 
                         NULL, 
                         sys_timer_post_callback);   

    xTimerStart(TIMER, 0); 
}
