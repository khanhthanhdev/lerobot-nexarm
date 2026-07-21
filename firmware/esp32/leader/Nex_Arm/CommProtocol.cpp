#include "CommProtocol.h"

static const char* TAG = "user_task";

static  PacketTypeDef g_rx_packet;

static uint8_t checksum_crc8(const uint8_t *data, uint8_t len) {
    uint16_t temp = 0;
    for (int i = 0; i < len; ++i) {
        temp += data[i];
    }
    return (uint8_t)(~temp);
}

void CommProtocol_t::begin()
{
    error_state = ERROR_NULL;
    parsing_state = PARSING_HEADER_1;
    successCallback = nullptr;
    errorCallback = nullptr;
}

void CommProtocol_t::register_success_callback(ProtocolSuccessCallback cb)
{
    successCallback = cb;
}

void CommProtocol_t::register_error_callback(ProtocolErrorCallback cb)
{
    errorCallback = cb;
}

uint8_t CommProtocol_t::tx_packet_complete(uint8_t id, uint8_t cmd, uint8_t* data, uint8_t data_len)
{
	uint8_t frame_len =  6 + data_len;  // 6: header1 + header2 + id + length + cmd + check
	
    tx_packet.header_1 = FRAME_HEADER_1;
    tx_packet.header_2 = FRAME_HEADER_2;
    tx_packet.elements.id = id;
    tx_packet.elements.length = 2 + data_len;
    tx_packet.elements.cmd = cmd;

	for(uint8_t i = 0; i < data_len; i++) {
		tx_packet.elements.args[i] = data[i];
	}
	
	tx_packet.elements.args[data_len] = checksum_crc8(tx_packet.data_raw, rx_packet.elements.length + 1);
    return frame_len;
}

void CommProtocol_t::parsing(uint8_t *data, uint16_t len)
{
    uint8_t arg_count = 0;
    uint8_t rec_count = 0;
    uint8_t checksum = 0;

    error_state = ERROR_NULL;
    parsing_state = PARSING_HEADER_1;

    for(uint8_t i = 0; i < len; i++) {
        ESP_LOGI(TAG, "%d |", data[i]);
    }

    ESP_LOGI(TAG, "\n");

    for (uint8_t i = 0; i < len; i++) {
        switch(parsing_state) {
            case PARSING_HEADER_1:
                if(data[rec_count] == FRAME_HEADER_1) {
                    rx_packet.header_1 = data[rec_count];
                    parsing_state = PARSING_HEADER_2;
                    ESP_LOGI(TAG, "H1:%d\n", rx_packet.header_1);
                }
                else {
                    error_state = ERROR_FRAME_HEADER;
                    ESP_LOGI(TAG, "E\n");
                }
                break;

            case PARSING_HEADER_2:
                if(data[rec_count] == FRAME_HEADER_2) {
                    rx_packet.header_2 = data[rec_count];
                    parsing_state = PARSING_ID;
                    ESP_LOGI(TAG, "H2:%d\n", rx_packet.header_2);
                }
                else {
                    parsing_state = PARSING_HEADER_1;
                    error_state = ERROR_FRAME_HEADER;
                    ESP_LOGI(TAG, "E\n");
                    if(errorCallback) {
                        errorCallback();
                    }  
                }
                break;

            case PARSING_ID: 
                rx_packet.elements.id = data[rec_count];
                parsing_state = PARSING_DATA_LENGTH;
                ESP_LOGI(TAG, "ID:%d\n", rx_packet.elements.id);
                break;

            case PARSING_DATA_LENGTH:
                if(data[rec_count] < (len - rec_count) ) {
                    rx_packet.elements.length = data[rec_count];
                    parsing_state = PARSING_CMD;
                    ESP_LOGI(TAG, "LENGTH:%d\n", rx_packet.elements.length);
                }
                else {
                    parsing_state = PARSING_HEADER_1;
                    error_state = ERROR_FRAME_LEN;
                    ESP_LOGI(TAG, "E\n");
                    if(errorCallback) {
                        errorCallback();
                    }  
                }
                break;

            case PARSING_CMD:
                rx_packet.elements.cmd = data[rec_count];
                if(rx_packet.elements.length == 2) {
                    parsing_state = PARSING_CHECKSUM;
                }
                else {
                    parsing_state = PARSING_ARGS;
                }
                ESP_LOGI(TAG, "CMD:%d\n", rx_packet.elements.cmd);
                break;

            case PARSING_ARGS:
                rx_packet.elements.args[arg_count] = data[rec_count];
                ESP_LOGI(TAG, "ARG:%d\n", rx_packet.elements.args[arg_count]);
                arg_count++;
                if(arg_count == rx_packet.elements.length - 2) {
                    arg_count = 0;
                    parsing_state = PARSING_CHECKSUM;
                }                    
                break;

            case PARSING_CHECKSUM: 
                checksum = checksum_crc8(rx_packet.data_raw, rx_packet.elements.length + 1);
                ESP_LOGI(TAG, "data: %d, checksum: %d\n", data[rec_count], checksum);
                ESP_LOGI(TAG, "I: %d\n",i);
                if(checksum != data[rec_count]) {
                    error_state = ERROR_CHEAKSUM;
                    ESP_LOGI(TAG, "E\n");
                }
                else {
                    rx_packet.elements.args[rx_packet.elements.length - 2] = data[rec_count];
                    if(successCallback) {
                        successCallback(&rx_packet);
                    }   
                    ESP_LOGI(TAG, "ARG:%d\n", rx_packet.elements.args[rx_packet.elements.length - 2]);
                }
                parsing_state = PARSING_HEADER_1;
                break;

            default:
                break;
        }

        rec_count++;
    }

    if(error_state != ERROR_NULL) {
        ESP_LOGI(TAG, "ERR %d\n", error_state);
        if(errorCallback) {
            errorCallback();
        }  
    }
}
