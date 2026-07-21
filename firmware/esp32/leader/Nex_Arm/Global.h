#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include "Nex_Arm_Board.h"

/* 宏函数 获得A的低八位 */
#define GET_LOW_BYTE(A) ((uint8_t)(A))
/* 宏函数 获得A的高八位 */
#define GET_HIGH_BYTE(A) ((uint8_t)((A) >> 8))
/* 宏函数 将高低八位合成为十六位 */
#define BYTE_TO_HW(A, B) ((((uint16_t)(A)) << 8) | (uint8_t)(B))

extern HW_Board board;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif