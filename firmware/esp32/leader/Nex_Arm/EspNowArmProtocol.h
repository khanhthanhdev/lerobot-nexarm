#ifndef ESP_NOW_ARM_PROTOCOL_H
#define ESP_NOW_ARM_PROTOCOL_H

#include <Arduino.h>

#define ARM_PROTO_VERSION  1

#define ARM_JOINT_NUM      6

enum ArmMsgType : uint8_t {
    ARM_MSG_JOINT_CMD   = 0x01,   
    ARM_MSG_JOINT_STATE = 0x02,   
    ARM_MSG_PING        = 0x03,   
};

enum ArmRole : uint8_t {
    ARM_ROLE_MASTER = 0,
    ARM_ROLE_SLAVE  = 1,
};

typedef struct __attribute__((packed)) {
    int16_t angle_cdeg;  
    int16_t reserved;    
    uint8_t acc;         
} ArmJointData_t;

typedef struct __attribute__((packed)) {
    uint8_t  magic;         
    uint8_t  version;      
    uint8_t  msg_type;      
    uint8_t  role;          
    uint8_t  seq;           
    uint8_t  reserved;      

    uint32_t timestamp_ms;  

    ArmJointData_t joint[ARM_JOINT_NUM]; 

    uint8_t checksum;       
} EspNowArmPacket_t;

inline uint8_t arm_calc_checksum(const EspNowArmPacket_t &pkt) {
    const uint8_t *p = reinterpret_cast<const uint8_t *>(&pkt);
    uint8_t sum = 0;
    for (size_t i = 0; i < sizeof(EspNowArmPacket_t) - 1; ++i) {
        sum += p[i];
    }
    return ~sum;
}

inline bool arm_packet_valid(const EspNowArmPacket_t &pkt) {
    if (pkt.magic != 0xA5) return false;
    if (pkt.version != ARM_PROTO_VERSION) return false;
    return pkt.checksum == arm_calc_checksum(pkt);
}

#endif 
