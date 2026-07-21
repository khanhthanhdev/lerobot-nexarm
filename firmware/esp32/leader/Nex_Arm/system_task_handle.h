#ifndef __SYSTEM_TASK_HANDLE_H__
#define __SYSTEM_TASK_HANDLE_H__

#include "esp_event.h"

#define FIRMWARE_VERSION            1

#define TIMER_PERIOD                20       //unit: ms

enum TimingEvent{
    TIMING_EVENT_BUZZER_UPDATE =    0,
    TIMING_EVENT_OLED_UPDATE,
    TIMING_EVENT_SERVO_STATUS_UPDATE,
    TIMING_EVENT_BAT_UPDATE,
};

enum TimingEventPeriod{  
    BUZZER_UPDATE_PERIOD =          20,  
    OLED_UPDATE_PERIOD  =           100,  
    SERVO_STATUS_UPDATE_PERIOD =    200,
    BAT_UPDATE_PERIOD =             200,
};

enum StatusEvent{
    STATUS_EVENT_LOOP =         0,
    STATUS_EVENT_USER_BUTTON,
    STATUS_EVENT_BOOT_BUTTON,
    STATUS_EVENT_ERROR,
    STATUS_EVENT_MAX
};

enum CtrlMode {
    UART_MODE = 0,  
    WIFI_MODE,
    BLE_MODE,
    CUSTOM_MODE
};

enum ProtocolCmd {
    CMD_FIRMWARE_VERSION_CHECK = 1,
    CMD_CHECK_BAT_LEVEL_CHECK = 2,
    CMD_ACTION_GROUP_RUN = 3,
    CMD_ACTION_GROUP_STOP = 4,
    CMD_ACTION_GROUP_DOWNLOAD = 5,
    CMD_FKINE_RESULT_GET = 6,
    CMD_IKINE_RESULT_GET = 7,
    CMD_COORDINATE_SET = 8
};

#ifdef __cplusplus
extern "C" {
#endif

void register_system_task(esp_event_loop_handle_t *event_loop);
void system_loop_handler(void);

#ifdef __cplusplus
}
#endif


#endif