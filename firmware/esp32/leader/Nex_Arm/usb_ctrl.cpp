#include "usb_ctrl.h"
#include "CommProtocol.h"

static const char* TAG = "user_ctrl";

static MessageBufferHandle_t  xSendBuffer;
static MessageBufferHandle_t  xReceiveBuffer;

SerialPort_t serial_port;

void SerialEvent()
{
    uint8_t size = serial_port.uart->available();
    uint8_t buff[(const uint8_t)size] = {0};

    serial_port.uart->readBytes(buff, size);
    
    xMessageBufferSend(xReceiveBuffer,
                      buff, // 要发送的数据地址
                      size, // 发送 1 个字节
                      0);
}

void SerialPort_t::begin(HardwareSerial& uart, uint32_t baudrate, uint8_t tx_pin, uint8_t rx_pin)
{
    this->uart = &uart;
    this->uart->setTxBufferSize(512);
    this->uart->setRxBufferSize(512);
    xReceiveBuffer = xMessageBufferCreate(MAX_MSG_BUF_SIZE);
    this->uart->onReceive(SerialEvent, true);
    this->uart->begin(baudrate, SERIAL_8N1, rx_pin, tx_pin);
    protocol.begin();
}

void SerialPort_t::register_ops_callback(ProtocolSuccessCallback cb)
{   
    protocol.register_success_callback(cb);
}

void SerialPort_t::rec_handler(void)
{
    real_msg_rec_size = xMessageBufferReceive(xReceiveBuffer, &rec_buffer, sizeof(rec_buffer), 0);
    if(real_msg_rec_size > 0) {
        protocol.parsing(rec_buffer, real_msg_rec_size);
        ESP_LOGI(TAG, "real read size :%d\n", real_msg_rec_size);
    }
}
