# -*- coding: utf-8 -*-
"""
K230 通信协议公共模块

包含 UART 和 I2C 通信共享的协议常量、编解码工具和基类。

协议格式:
- 帧头: 0xAA 0x55 (2字节)
- 帧结构: Header(2) + Len(2) + Ctrl(1) + Func(1) + Txn(1) + Payload + XOR(1)
- Ctrl字段: Bit7-6=帧类型, Bit5=续包标志, Bit4-0=分包序号(0-31)
- Txn字段: 事务号(1-255)，用于匹配命令响应；主动上报固定为 0
"""

import struct
import time

# ============================================================================
# 协议常量
# ============================================================================

FRAME_HEADER = bytes([0xAA, 0x55])
FRAME_HEADER_0 = 0xAA
FRAME_HEADER_1 = 0x55

FRAME_LEN_OFFSET = 2
FRAME_CTRL_OFFSET = 4
FRAME_SEQ_OFFSET = FRAME_CTRL_OFFSET
FRAME_FUNC_OFFSET = 5
FRAME_TXN_OFFSET = 6
FRAME_PAYLOAD_OFFSET = 7

MIN_FRAME_LEN = 8  # Header(2) + Len(2) + Ctrl(1) + Func(1) + Txn(1) + XOR(1)
MAX_FRAME_LEN = 4096  # 最大帧长度
MAX_PAYLOAD_LEN = MAX_FRAME_LEN - MIN_FRAME_LEN  # 最大载荷长度

# SEQ字段掩码
TYPE_MASK = 0xC0  # 高2位: 帧类型
CONT_MASK = 0x20  # Bit5: 续包标志
SEQ_MASK = 0x1F  # 低5位: 序列号(0-31)

# 帧类型 (SEQ字段高2位)
FRAME_TYPE_CMD = 0x00  # 命令帧（外部控制器→K230）
FRAME_TYPE_RSP = 0x40  # 响应帧（K230→外部控制器）
FRAME_TYPE_RPT = 0x80  # 上报帧（K230主动上报）
FRAME_TYPE_ACK = 0xC0  # 确认/否定帧

PROTOCOL_VERSION_MAJOR = 2
PROTOCOL_VERSION_MINOR = 6

PROTOCOL_CAP_RSP_FRAMES = 1 << 0
PROTOCOL_CAP_I2C_MAILBOX_V2 = 1 << 1
PROTOCOL_CAP_FRAGMENT_SEQ_CHECK = 1 << 2
PROTOCOL_CAP_PROTOCOL_INFO = 1 << 3
PROTOCOL_CAP_HEARTBEAT_NO_RESULT_DATA = 1 << 4
PROTOCOL_CAP_ERROR_DETAIL = 1 << 5
PROTOCOL_CAP_IDLE_HEARTBEAT = 1 << 6

PROTOCOL_CAP_FLAGS = (
    PROTOCOL_CAP_RSP_FRAMES |
    PROTOCOL_CAP_I2C_MAILBOX_V2 |
    PROTOCOL_CAP_FRAGMENT_SEQ_CHECK |
    PROTOCOL_CAP_PROTOCOL_INFO |
    PROTOCOL_CAP_HEARTBEAT_NO_RESULT_DATA |
    PROTOCOL_CAP_ERROR_DETAIL |
    PROTOCOL_CAP_IDLE_HEARTBEAT
)

# ============================================================================
# 功能码定义 - 系统控制 (0x01-0x0F)
# ============================================================================
CMD_SET_MODE = 0x01
CMD_SET_VOLUME = 0x02
CMD_SET_WIFI = 0x03
CMD_REQUEST_STATUS = 0x04  # 请求设备立即上报一次最新状态心跳
CMD_CLEAR_MEMORY = 0x05    # 清空 wonderlens_system 结果/异步状态缓存
CMD_GET_PROTOCOL_INFO = 0x06  # 获取协议版本/能力/当前传输帧能力

# ============================================================================
# 功能码定义 - 检测参数 (0x10-0x1F)
# ============================================================================
CMD_SET_CONF_THRESH = 0x10
CMD_SET_NMS_THRESH = 0x11
CMD_SEG_SET_MASK_THRESH = 0x12
CMD_SET_SIMPLE_RESULT = 0x13
CMD_DISABLE_RUN = 0x14

# ============================================================================
# 功能码定义 - 人脸/人体关键点/手掌关键点 (0x20-0x3F)
# ============================================================================
CMD_FACE_LEARN = 0x20
CMD_FACE_DELETE = 0x21
CMD_FACE_RENAME = 0x22
CMD_FACE_SET_RECOG_CONF = 0x23
CMD_FACE_HIGH_PRECISION = 0x24
CMD_FACE_ENABLE_KEYPOINT = 0x25  # 启用关键点 uint8
CMD_FACE_DETECT_ONLY = 0x26     # 只检测人脸 uint8
CMD_FACE_ENHANCE_LEARN = 0x27  # 人脸加强学习 字符串名称
CMD_FACE_LEARN_AT_POINT = 0x28  # 人脸按坐标学习 uint32 x + uint32 y + 字符串名称
CMD_FACE_SET_POSE_THRESH = 0x29  # 人脸姿态/注视匹配阈值 uint8 roll + uint8 pitch + uint8 yaw
CMD_PERSON_KP_LEARN = 0x2A
CMD_PERSON_KP_DELETE = 0x2B
CMD_PERSON_KP_RENAME = 0x2C
CMD_PERSON_KP_ENHANCE_LEARN = 0x2D  # 人体关键点加强学习 字符串名称
CMD_HAND_KP_LEARN = 0x2E
CMD_HAND_KP_DELETE = 0x2F
CMD_HAND_KP_RENAME = 0x30
CMD_HAND_KP_ENHANCE_LEARN = 0x31  # 手掌关键点加强学习 字符串名称
CMD_HAND_DETECT_ONLY = 0x32  # 只检测手掌 uint8

# ============================================================================
# 功能码定义 - 颜色检测 (0x40-0x4F)
# ============================================================================
CMD_COLOR_SET_TARGET = 0x40
CMD_COLOR_SET_THRESH = 0x41
CMD_COLOR_GET_THRESH = 0x42
CMD_COLOR_SET_FILTER = 0x43
CMD_COLOR_SET_MIN_AREA = 0x44
CMD_MULTI_COLOR_SET_LIST = 0x45
CMD_LINE_SET_ROI = 0x46
CMD_COLOR_LEARNING_SET_POINT = 0x47
CMD_COLOR_LEARNING_SAVE = 0x48
CMD_COLOR_LEARNING_RENAME = 0x49
CMD_COLOR_LEARNING_DELETE = 0x4A

# ============================================================================
# 功能码定义 - 跟踪/动态手势/自学习/物体/自定义 (0x50-0x5F)
# ============================================================================
CMD_NANOTRACK_SET_RECT = 0x50
CMD_NANOTRACK_STOP = 0x51
CMD_GESTURE_SET_FRAME = 0x52
CMD_DGESTURE_CTRL = 0x53  # 动态手势控制 uint8 action + [name] + [name2]
CMD_DGESTURE_ENHANCE_SAVE = 0x54  # 动态手势加强保存 字符串名称
CMD_SELFLEARN_SET_NAME = 0x55
CMD_SELFLEARN_SET_RECT = 0x56
CMD_SELFLEARN_SET_FRAME = 0x57
CMD_SELFLEARN_SET_FEATURES = 0x58
CMD_SELFLEARN_DELETE = 0x59
CMD_SELFLEARN_RENAME = 0x5A  # 自学习重命名 old_name + new_name
CMD_OBJECT_SET_MODE = 0x5B  # 物体识别模式切换 uint8(0:detect,1:cls)
CMD_CUSTOM_SET_MODEL = 0x5C

DGESTURE_ACTION_NONE = 0
DGESTURE_ACTION_RECORD_START = 1
DGESTURE_ACTION_RECORD_STOP = 2
DGESTURE_ACTION_SAVE = 3
DGESTURE_ACTION_DELETE = 4
DGESTURE_ACTION_RENAME = 5
DGESTURE_ACTION_SAVE_APPEND = 6
DGESTURE_ACTION_SAVE_APPEND_DROP_OLDEST = 7

# ============================================================================
# 功能码定义 - 语音交互/媒体 (0x60-0x86)
# ============================================================================
CMD_SET_LLM_KEY = 0x60        # 设置LLM密钥 字符串参数
CMD_SET_TTS_VOICE = 0x61      # 设置TTS音色 两个字符串参数
CMD_SET_ASR_LANG = 0x62       # 设置ASR语言 字符串参数
CMD_SET_THINKING = 0x63       # 启用思考模式 uint8
CMD_SET_SEARCH = 0x64         # 启用搜索模式 uint8
CMD_SET_START_SILENCE = 0x65  # 设置开始语句静音时间
CMD_SET_END_SILENCE = 0x66    # 设置结束语句静音时间
CMD_SET_PROMPT = 0x67         # 设置系统提示词 字符串参数
CMD_ASR = 0x68                # 开启语音识别 uint8
CMD_TTS = 0x69                # 文本转语音 字符串参数
CMD_LLM_CHAT = 0x6A           # LLM对话 字符串参数
CMD_VLM_CHAT = 0x6B           # VLM对话 字符串参数
CMD_SET_MCP_TOOLS = 0x6C      # 设置MCP工具
CMD_RESULT_RETURN = 0x6D      # 返回结果,包含mcp,asr,llm等
CMD_EMPTY_RETURN = 0x6E       # 空返回，常见于 WiFi 连接结果或 TTS 完成通知
CMD_SET_LLM_MODEL = 0x6F      # 设置LLM模型 字符串参数
CMD_SET_VLM_MODEL = 0x70      # 设置VLM模型 字符串参数
CMD_SET_LLM_BASE_URL = 0x71   # 设置LLM服务地址 字符串参数
CMD_SET_VLM_BASE_URL = 0x72   # 设置VLM服务地址 字符串参数
CMD_SET_SPEECH_URL = 0x73     # 设置语音服务地址 字符串参数
CMD_MEDIA_CAMERA_SNAPSHOT = 0x81  # 媒体相机拍照
CMD_MEDIA_SET_PHOTO_PREFIX = 0x82     # 设置照片名前缀 字符串参数，可为空
CMD_MEDIA_DELETE_PHOTO = 0x83         # 删除指定照片 字符串参数
CMD_MEDIA_ENTER_CAMERA_APP = 0x85     # 进入媒体相机 app
CMD_MEDIA_SET_PHOTO_START = 0x86      # 设置拍照命名起始值 uint32
CMD_ASR_TIMEOUT_MS = 70000

# ============================================================================
# 功能码定义 - 状态上报 (0x70-0x7F)
# ============================================================================
RPT_HEARTBEAT = 0x70
RPT_ERROR = 0x71
RPT_DETECT_BBOX = 0x72
RPT_DETECT_STR = 0x73
RPT_DETECT_OCR = 0x74
RPT_DETECT_COLOR = 0x75
RPT_DETECT_LINE = 0x76
RPT_DETECT_KEYPOINT = 0x77
RPT_DETECT_HAND_KP = 0x78
RPT_DETECT_CENTER = 0x79
RPT_DETECT_FACE_KP = 0x7A
RPT_DETECT_QUAD = 0x7B

FRAME_TYPE_NAMES = {
    FRAME_TYPE_CMD: 'cmd',
    FRAME_TYPE_RSP: 'rsp',
    FRAME_TYPE_RPT: 'rpt',
    FRAME_TYPE_ACK: 'ack',
}

REPORT_KIND_NAMES = {
    RPT_HEARTBEAT: 'heartbeat',
    RPT_ERROR: 'error',
    RPT_DETECT_BBOX: 'bbox',
    RPT_DETECT_STR: 'string',
    RPT_DETECT_OCR: 'ocr',
    RPT_DETECT_COLOR: 'color',
    RPT_DETECT_LINE: 'line',
    RPT_DETECT_KEYPOINT: 'keypoint',
    RPT_DETECT_HAND_KP: 'hand_kp',
    RPT_DETECT_CENTER: 'center',
    RPT_DETECT_FACE_KP: 'face_kp',
    RPT_DETECT_QUAD: 'quad',
}

# ============================================================================
# 错误码定义
# ============================================================================
ERR_OK = 0x00
ERR_UNKNOWN_CMD = 0x01
ERR_INVALID_MODE = 0x02
ERR_INVALID_PARAM = 0x03
ERR_DATA_LEN = 0x04
ERR_BUSY = 0x05
ERR_NOT_READY = 0x06
ERR_BUFFER_FULL = 0x07
ERR_EXEC_FAIL = 0x08
ERR_FRAME_INVALID = 0x09
ERR_XOR_FAIL = 0x0A
ERR_SEQ_MISMATCH = 0x0B
ERR_REASSEMBLE_FAIL = 0x0C
ERR_CALLBACK_FAIL = 0x0D
ERR_UART_WRITE = 0x0E
ERR_UART_READ = 0x0F
RSP_ERROR_PREFIX_LEN = 4

ERR_MODULE_NONE = 0x00
ERR_MODULE_PROTOCOL = 0x01
ERR_MODULE_TRANSPORT = 0x02
ERR_MODULE_COMMAND = 0x03
ERR_MODULE_RUNTIME = 0x04
ERR_MODULE_IPC = 0x05
ERR_MODULE_REPORT = 0x06
ERR_MODULE_SPEECH = 0x07
ERR_MODULE_MEDIA = 0x08
ERR_MODULE_UNKNOWN = 0xFF

ERROR_NAMES = {
    ERR_OK: "成功",
    ERR_UNKNOWN_CMD: "未知命令",
    ERR_INVALID_MODE: "无效模式",
    ERR_INVALID_PARAM: "参数无效",
    ERR_DATA_LEN: "数据长度错误",
    ERR_BUSY: "系统忙",
    ERR_NOT_READY: "未就绪",
    ERR_BUFFER_FULL: "缓冲区满",
    ERR_EXEC_FAIL: "执行失败",
    ERR_FRAME_INVALID: "帧无效",
    ERR_XOR_FAIL: "XOR校验失败",
    ERR_SEQ_MISMATCH: "分包序列不匹配",
    ERR_REASSEMBLE_FAIL: "重组失败",
    ERR_CALLBACK_FAIL: "回调失败",
    ERR_UART_WRITE: "串口写失败",
    ERR_UART_READ: "串口读失败",
}

ERROR_MODULE_NAMES = {
    ERR_MODULE_NONE: "none",
    ERR_MODULE_PROTOCOL: "protocol",
    ERR_MODULE_TRANSPORT: "transport",
    ERR_MODULE_COMMAND: "command",
    ERR_MODULE_RUNTIME: "runtime",
    ERR_MODULE_IPC: "ipc",
    ERR_MODULE_REPORT: "report",
    ERR_MODULE_SPEECH: "speech",
    ERR_MODULE_MEDIA: "media",
    ERR_MODULE_UNKNOWN: "unknown",
}

# ============================================================================
# 应用名称映射
# ============================================================================
CANONICAL_APP_NAME_TO_INDEX = {
    'Empty': 0,
    'FaceDetection': 1,
    'FaceLandmark': 2,
    'FacePose': 3,
    'FaceRecognition': 4,
    'FaceParse': 5,
    'FaceMesh': 6,
    'PersonDetection': 7,
    'PersonKeypointDetect': 8,
    'HandDetection': 9,
    'HandRecognition': 10,
    'HandKeyPointDetection': 11,
    'HandGesture': 12,
    'FaceLiveness': 13,
    'FalldownDetection': 14,
    'EyeGaze': 15,
    'ObjectTrack': 16,
    'GarbageClassification': 17,
    'DynamicGesture': 18,
    'TrafficDetection': 19,
    'AiLLM_Mode': 20,
    'SingleColorDetection': 21,
    'MultiColorDetection': 22,
    'LineDetection': 23,
    'ColorTracking': 24,
    'OCRDetection': 25,
    'OCRRecognition': 26,
    'LicencePlateDetection': 27,
    'LicencePlateRecognition': 28,
    'ObjectDetection': 29,
    'Segmentation': 30,
    'SelfLearning': 31,
    'ApriltagDiscern': 32,
    'DMCodeDiscern': 33,
    'QRCodeDiscern': 34,
    'BarCodeDiscern': 35,
    'CustomDetection': 36,
}

APP_NAME_TO_INDEX = dict(CANONICAL_APP_NAME_TO_INDEX)

APP_INDEX_TO_NAME = {v: k for k, v in CANONICAL_APP_NAME_TO_INDEX.items()}
FACE_DB_MODE_NAMES = ('FaceRecognition', 'FacePose', 'EyeGaze')

RESULT_COORD_MAX_X = 320
RESULT_COORD_MAX_Y = 240
RESULT_KEYPOINT_MARGIN_X = max(64, RESULT_COORD_MAX_X // 2)
RESULT_KEYPOINT_MARGIN_Y = max(64, RESULT_COORD_MAX_Y // 2)
RESULT_KEYPOINT_MIN_X = -RESULT_KEYPOINT_MARGIN_X
RESULT_KEYPOINT_MAX_X = RESULT_COORD_MAX_X + RESULT_KEYPOINT_MARGIN_X
RESULT_KEYPOINT_MIN_Y = -RESULT_KEYPOINT_MARGIN_Y
RESULT_KEYPOINT_MAX_Y = RESULT_COORD_MAX_Y + RESULT_KEYPOINT_MARGIN_Y
RESULT_OCR_MARGIN_X = max(32, RESULT_COORD_MAX_X // 10)
RESULT_OCR_MARGIN_Y = max(24, RESULT_COORD_MAX_Y // 10)
RESULT_OCR_MIN_X = -RESULT_OCR_MARGIN_X
RESULT_OCR_MAX_X = RESULT_COORD_MAX_X + RESULT_OCR_MARGIN_X
RESULT_OCR_MIN_Y = -RESULT_OCR_MARGIN_Y
RESULT_OCR_MAX_Y = RESULT_COORD_MAX_Y + RESULT_OCR_MARGIN_Y
RESULT_EYE_GAZE_TARGET_MARGIN = max(64, RESULT_COORD_MAX_X // 2)
RESULT_EYE_GAZE_TARGET_MIN_X = -RESULT_EYE_GAZE_TARGET_MARGIN
RESULT_EYE_GAZE_TARGET_MAX_X = RESULT_COORD_MAX_X + RESULT_EYE_GAZE_TARGET_MARGIN
RESULT_EYE_GAZE_TARGET_MIN_Y = -RESULT_EYE_GAZE_TARGET_MARGIN
RESULT_EYE_GAZE_TARGET_MAX_Y = RESULT_COORD_MAX_Y + RESULT_EYE_GAZE_TARGET_MARGIN
RESULT_ANGLE_ABS_MAX = 360
RESULT_SCORE_MAX = 100
RESULT_LINE_ROI_COUNT = 3
MAX_RESULT_VALIDATION_ISSUES = 8

MODE_ALLOWED_RESULT_TYPES = {
    'FaceDetection': ('bbox', 'center'),
    'FaceLandmark': ('bbox', 'center'),
    'FacePose': ('bbox', 'center'),
    'FaceRecognition': ('bbox', 'center'),
    'FaceParse': ('bbox', 'center'),
    'FaceMesh': ('bbox', 'center'),
    'PersonDetection': ('bbox', 'center'),
    'PersonKeypointDetect': ('keypoint',),
    'HandDetection': ('bbox', 'center'),
    'HandRecognition': ('bbox', 'center'),
    'HandKeyPointDetection': ('bbox', 'center', 'hand_kp'),
    'HandGesture': ('bbox', 'center', 'hand_kp'),
    'FaceLiveness': ('bbox', 'center'),
    'FalldownDetection': ('bbox', 'center'),
    'EyeGaze': ('bbox', 'center'),
    'ObjectTrack': ('bbox', 'center'),
    'GarbageClassification': ('quad', 'center'),
    'DynamicGesture': ('string', 'string_list'),
    'TrafficDetection': ('bbox', 'center'),
    'SingleColorDetection': ('multi_color', 'center'),
    'MultiColorDetection': ('multi_color', 'center'),
    'LineDetection': ('line', 'center'),
    'ColorTracking': ('multi_color', 'center'),
    'OCRDetection': ('quad', 'center'),
    'OCRRecognition': ('ocr', 'center'),
    'LicencePlateDetection': ('bbox', 'center'),
    'LicencePlateRecognition': ('ocr', 'center'),
    'ObjectDetection': ('bbox', 'center'),
    'Segmentation': ('bbox', 'center'),
    'SelfLearning': ('bbox', 'string_list'),
    'ApriltagDiscern': ('quad', 'center'),
    'DMCodeDiscern': ('quad', 'center'),
    'QRCodeDiscern': ('bbox', 'string_list'),
    'BarCodeDiscern': ('bbox', 'string_list'),
    'CustomDetection': ('bbox', 'quad', 'center'),
}

FACE_EMOTION_LABELS_EN = (
    'Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'
)
FACE_EMOTION_LABELS_ZH = (
    '愤怒', '厌恶', '害怕', '开心', '中性', '难过', '惊讶'
)

HAND_GESTURE_LABELS = frozenset((
    'ok', 'fist', 'five', 'gun', 'love', 'one', 'six', 'three',
    'thumbUp', 'yeah', 'other', 'unknown'
))

GARBAGE_CLASS_LABELS = frozenset((
    'BananaPeel', 'BrokenBones', 'CigaretteEnd', 'DisposableChopsticks',
    'Ketchup', 'Marker', 'OralLiquidBottle', 'Plate',
    'PlasticBottle', 'StorageBattery', 'Toothbrush', 'Umbrella',
))

TRAFFIC_LABELS = frozenset((
    'red_barrier', 'go_straight', 'turn_left', 'turn_right',
    'roundabout', 'parking_area', 'stop_sign',
    'traffic_light_red', 'traffic_light_yellow', 'traffic_light_green',
    'pedestrian_crossing',
))

# ============================================================================
# 数据类型标签
# ============================================================================
TYPE_NULL = 0x00
TYPE_BOOL_F = 0x01
TYPE_BOOL_T = 0x02
TYPE_INT8 = 0x03
TYPE_INT16 = 0x04
TYPE_INT32 = 0x05
TYPE_UINT8 = 0x06
TYPE_UINT16 = 0x07
TYPE_UINT32 = 0x08
TYPE_STRING = 0x09
TYPE_ARRAY = 0x0A
TYPE_DICT = 0x0B
TYPE_VARINT = 0x0C


# ============================================================================
# XOR校验计算
# ============================================================================
def calc_xor(data, offset=0, length=None):
    """计算XOR校验值"""
    if length is None:
        length = len(data) - offset
    xor_val = 0
    for i in range(offset, offset + length):
        xor_val ^= data[i]
    return xor_val


# ============================================================================
# 数据序列化
# ============================================================================
def data_pack(obj):
    """将Python对象序列化为二进制格式"""
    buf = bytearray()

    if obj is None:
        buf.append(TYPE_NULL)
    elif isinstance(obj, bool):
        buf.append(TYPE_BOOL_T if obj else TYPE_BOOL_F)
    elif isinstance(obj, int):
        if 0 <= obj <= 255:
            buf.append(TYPE_UINT8)
            buf.append(obj)
        elif -128 <= obj <= 127:
            buf.append(TYPE_INT8)
            buf.append(obj & 0xFF)
        elif 0 <= obj <= 65535:
            buf.append(TYPE_UINT16)
            buf.extend(struct.pack('>H', obj))
        elif -32768 <= obj <= 32767:
            buf.append(TYPE_INT16)
            buf.extend(struct.pack('>h', obj))
        elif 0 <= obj <= 0xFFFFFFFF:
            buf.append(TYPE_UINT32)
            buf.extend(struct.pack('>I', obj))
        elif -2147483648 <= obj <= 2147483647:
            buf.append(TYPE_INT32)
            buf.extend(struct.pack('>i', obj))
        else:
            raise ValueError("Integer out of range: %s" % obj)
    elif isinstance(obj, str):
        utf8_bytes = obj.encode('utf-8')
        if len(utf8_bytes) > 65535:
            raise ValueError("String too long")
        buf.append(TYPE_STRING)
        buf.extend(struct.pack('>H', len(utf8_bytes)))
        buf.extend(utf8_bytes)
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 65535:
            raise ValueError("Array too long")
        buf.append(TYPE_ARRAY)
        buf.extend(struct.pack('>H', len(obj)))
        for item in obj:
            buf.extend(data_pack(item))
    elif isinstance(obj, dict):
        if len(obj) > 65535:
            raise ValueError("Dict too large")
        buf.append(TYPE_DICT)
        buf.extend(struct.pack('>H', len(obj)))
        for k, v in obj.items():
            buf.extend(data_pack(k))
            buf.extend(data_pack(v))
    else:
        raise TypeError("Unsupported type: %s" % type(obj))

    return bytes(buf)


def data_unpack(data, offset=0):
    """将二进制格式反序列化为Python对象"""
    if offset >= len(data):
        raise ValueError("Data too short")

    type_tag = data[offset]
    consumed = 1

    if type_tag == TYPE_NULL:
        return None, consumed
    elif type_tag == TYPE_BOOL_F:
        return False, consumed
    elif type_tag == TYPE_BOOL_T:
        return True, consumed
    elif type_tag == TYPE_INT8:
        val = struct.unpack_from('>b', data, offset + 1)[0]
        return val, consumed + 1
    elif type_tag == TYPE_INT16:
        val = struct.unpack_from('>h', data, offset + 1)[0]
        return val, consumed + 2
    elif type_tag == TYPE_INT32:
        val = struct.unpack_from('>i', data, offset + 1)[0]
        return val, consumed + 4
    elif type_tag == TYPE_UINT8:
        val = data[offset + 1]
        return val, consumed + 1
    elif type_tag == TYPE_UINT16:
        val = struct.unpack_from('>H', data, offset + 1)[0]
        return val, consumed + 2
    elif type_tag == TYPE_UINT32:
        val = struct.unpack_from('>I', data, offset + 1)[0]
        return val, consumed + 4
    elif type_tag == TYPE_STRING:
        str_len = struct.unpack_from('>H', data, offset + 1)[0]
        consumed += 2
        str_bytes = data[offset + consumed:offset + consumed + str_len]
        return str_bytes.decode('utf-8'), consumed + str_len
    elif type_tag == TYPE_ARRAY:
        arr_len = struct.unpack_from('>H', data, offset + 1)[0]
        consumed += 2
        result = []
        for _ in range(arr_len):
            item, item_consumed = data_unpack(data, offset + consumed)
            result.append(item)
            consumed += item_consumed
        return result, consumed
    elif type_tag == TYPE_DICT:
        dict_len = struct.unpack_from('>H', data, offset + 1)[0]
        consumed += 2
        result = {}
        for _ in range(dict_len):
            key, key_consumed = data_unpack(data, offset + consumed)
            consumed += key_consumed
            val, val_consumed = data_unpack(data, offset + consumed)
            consumed += val_consumed
            result[key] = val
        return result, consumed
    else:
        raise ValueError("Unknown type tag: 0x%02X" % type_tag)


# ============================================================================
# 紧凑编码器
# ============================================================================
class CompactCodec:
    """紧凑编码器"""

    @staticmethod
    def encode_string(s):
        """编码字符串: [长度1字节][内容N字节]"""
        if s is None:
            return bytes([0])
        data = s.encode('utf-8')
        if len(data) > 255:
            data = data[:255]
        return bytes([len(data)]) + data

    @staticmethod
    def decode_string(data, offset=0):
        """解码字符串 -> (字符串, 消耗字节数)"""
        if offset >= len(data):
            return "", 1
        length = data[offset]
        if offset + 1 + length > len(data):
            return "", 1
        content = data[offset + 1:offset + 1 + length]
        return content.decode('utf-8', 'ignore'), 1 + length

    @staticmethod
    def encode_uint8(value):
        """编码uint8"""
        if value is None:
            return bytes([0])
        return bytes([value & 0xFF])

    @staticmethod
    def encode_uint16(value):
        """编码uint16大端序"""
        if value is None:
            return struct.pack('>H', 0)
        return struct.pack('>H', value & 0xFFFF)

    @staticmethod
    def encode_uint32(value):
        """编码uint32大端序"""
        if value is None:
            return struct.pack('>I', 0)
        return struct.pack('>I', value & 0xFFFFFFFF)

    @staticmethod
    def encode_bbox(x, y, w, h):
        """编码边界框: 8字节大端序"""
        return struct.pack('>HHHH', x & 0xFFFF, y & 0xFFFF, w & 0xFFFF, h & 0xFFFF)

    @staticmethod
    def decode_bbox(data, offset=0):
        """解码边界框 -> ((x, y, w, h), 消耗字节数)"""
        if offset + 8 > len(data):
            return (0, 0, 0, 0), 8
        x, y, w, h = struct.unpack_from('>HHHH', data, offset)
        return (x, y, w, h), 8

    @staticmethod
    def encode_lab_thresh(l_min, l_max, a_min, a_max, b_min, b_max):
        """编码LAB阈值: 6字节"""

        def to_ubyte(val):
            return val & 0xFF

        return bytes([to_ubyte(l_min), to_ubyte(l_max),
                      to_ubyte(a_min), to_ubyte(a_max),
                      to_ubyte(b_min), to_ubyte(b_max)])

    @staticmethod
    def decode_lab_thresh(data, offset=0):
        """解码LAB阈值"""
        if offset + 6 > len(data):
            return (0, 0, 0, 0, 0, 0), 6
        return tuple(data[offset:offset + 6]), 6

    @staticmethod
    def encode_line_roi(roi_list):
        """编码线检测ROI: 3个区域，每个5个百分比字节，共15字节"""
        if len(roi_list) != 3:
            raise ValueError("Line ROI must have exactly 3 regions")
        buf = bytearray()
        for x, y, w, h, weight in roi_list:
            values = []
            for value in (x, y, w, h, weight):
                int_value = int(round(value))
                if int_value < 0 or int_value > 100:
                    raise ValueError("Line ROI value must be in range 0..100")
                values.append(int_value)
            buf.extend(values)
        return bytes(buf)


# ============================================================================
# 帧数据类
# ============================================================================
class UARTFrame:
    """UART帧数据结构"""

    def __init__(self):
        self.frame_type = FRAME_TYPE_CMD
        self.continuation = False
        self.sequence = 0
        self.func_code = 0
        self.txn_id = 0
        self.payload = b''

    def get_seq_byte(self):
        seq_byte = self.frame_type & TYPE_MASK
        if self.continuation:
            seq_byte |= CONT_MASK
        seq_byte |= (self.sequence & SEQ_MASK)
        return seq_byte

    def set_from_seq_byte(self, seq_byte):
        self.frame_type = seq_byte & TYPE_MASK
        self.continuation = bool(seq_byte & CONT_MASK)
        self.sequence = seq_byte & SEQ_MASK

    def __repr__(self):
        type_names = {
            FRAME_TYPE_CMD: 'CMD', FRAME_TYPE_RSP: 'RSP',
            FRAME_TYPE_RPT: 'RPT', FRAME_TYPE_ACK: 'ACK'
        }
        return "UARTFrame(type=%s, seq=%s, txn=%s, func=0x%02X)" % (
            type_names.get(self.frame_type, 'UNK'),
            self.sequence,
            self.txn_id,
            self.func_code,
        )


# ============================================================================
# 帧构建
# ============================================================================
def build_frame(frame_type, sequence, func_code, payload=b'', continuation=False, txn_id=0):
    """构建完整的UART帧"""
    if len(payload) > MAX_PAYLOAD_LEN:
        payload = payload[:MAX_PAYLOAD_LEN]

    seq_byte = (frame_type & TYPE_MASK)
    if continuation:
        seq_byte |= CONT_MASK
    seq_byte |= (sequence & SEQ_MASK)

    frame = bytearray()
    frame.extend(FRAME_HEADER)
    frame.extend(struct.pack('>H', len(payload)))
    frame.append(seq_byte)
    frame.append(func_code)
    frame.append(txn_id & 0xFF)
    frame.extend(payload)
    xor_val = calc_xor(frame, FRAME_LEN_OFFSET, len(frame) - FRAME_LEN_OFFSET)
    frame.append(xor_val)
    return bytes(frame)


def split_payload(data, max_size=MAX_PAYLOAD_LEN):
    """将大数据分割为多个分包"""
    if not data:
        return [(b'', False)]
    chunks = []
    offset = 0
    while offset < len(data):
        chunk_size = min(max_size, len(data) - offset)
        chunk = data[offset:offset + chunk_size]
        offset += chunk_size
        chunks.append((chunk, offset < len(data)))
    return chunks


# ============================================================================
# 帧解析器
# ============================================================================
class FrameParser:
    """帧解析器，支持流式解析和帧同步"""

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data):
        if data:
            self.buffer.extend(data)

    def parse_one(self):
        # 搜索帧头
        while len(self.buffer) >= 2:
            if self.buffer[0] == FRAME_HEADER_0 and self.buffer[1] == FRAME_HEADER_1:
                break
            self.buffer = self.buffer[1:]

        if len(self.buffer) < MIN_FRAME_LEN:
            return None

        payload_len = struct.unpack_from('>H', self.buffer, FRAME_LEN_OFFSET)[0]
        if payload_len > MAX_PAYLOAD_LEN:
            self.buffer = self.buffer[2:]
            return None

        frame_len = MIN_FRAME_LEN + payload_len
        if len(self.buffer) < frame_len:
            return None

        frame_data = bytes(self.buffer[:frame_len])
        calc_xor_val = calc_xor(frame_data, FRAME_LEN_OFFSET, frame_len - FRAME_LEN_OFFSET - 1)
        recv_xor_val = frame_data[-1]

        if calc_xor_val != recv_xor_val:
            self.buffer = self.buffer[2:]
            return None

        frame = UARTFrame()
        frame.set_from_seq_byte(frame_data[FRAME_SEQ_OFFSET])
        frame.func_code = frame_data[FRAME_FUNC_OFFSET]
        frame.txn_id = frame_data[FRAME_TXN_OFFSET]
        frame.payload = frame_data[FRAME_PAYLOAD_OFFSET:-1] if payload_len > 0 else b''
        self.buffer = self.buffer[frame_len:]
        return frame

    def parse_all(self):
        frames = []
        while True:
            frame = self.parse_one()
            if frame is None:
                break
            frames.append(frame)
        return frames

    def clear(self):
        self.buffer = bytearray()


# ============================================================================
# 分包重组器
# ============================================================================
class PayloadReassembler:
    """分包重组器"""

    def __init__(self, timeout_ms=1000):
        self.timeout_ms = timeout_ms
        self.buffer = bytearray()
        self.frame_type = None
        self.func_code = None
        self.txn_id = None
        self.expected_sequence = None
        self.start_time = 0
        self.in_progress = False
        self._get_time_ms = None  # 由子类设置时间获取函数

    def set_time_func(self, func):
        """设置时间获取函数"""
        self._get_time_ms = func

    def _current_time_ms(self):
        if self._get_time_ms:
            return self._get_time_ms()
        return 0

    def feed(self, frame):
        current_time = self._current_time_ms()
        if self.in_progress and (current_time - self.start_time) > self.timeout_ms:
            self.reset()

        if not self.in_progress:
            self.buffer = bytearray()
            self.frame_type = frame.frame_type
            self.func_code = frame.func_code
            self.txn_id = frame.txn_id
            self.expected_sequence = frame.sequence
            self.start_time = current_time
            self.in_progress = True
        elif (
            frame.frame_type != self.frame_type or
            frame.func_code != self.func_code or
            frame.txn_id != self.txn_id or
            frame.sequence != self.expected_sequence
        ):
            self.reset()
            raise ValueError("fragment mismatch")

        self.buffer.extend(frame.payload)
        self.expected_sequence = (frame.sequence + 1) & SEQ_MASK

        if not frame.continuation:
            result = bytes(self.buffer)
            self.reset()
            return result
        return None

    def reset(self):
        self.buffer = bytearray()
        self.frame_type = None
        self.func_code = None
        self.txn_id = None
        self.expected_sequence = None
        self.start_time = 0
        self.in_progress = False


# ============================================================================
# 心跳数据类
# ============================================================================
class HeartbeatData:
    """心跳数据 - 紧凑格式"""

    STATUS_RUN = 0x01
    STATUS_RESULT = 0x02
    STATUS_READY = 0x04
    STATUS_BUSY = 0x08
    STATUS_ERR = 0x10

    def __init__(self, mode=0, status=0):
        self.mode = mode
        self.status = status

    @property
    def is_running(self):
        return bool(self.status & self.STATUS_RUN)

    @property
    def has_result(self):
        return bool(self.status & self.STATUS_RESULT)

    @property
    def is_ready(self):
        return bool(self.status & self.STATUS_READY)

    @property
    def is_busy(self):
        return bool(self.status & self.STATUS_BUSY)

    @property
    def has_error(self):
        return bool(self.status & self.STATUS_ERR)

    def encode(self):
        return bytes([self.mode, self.status])

    @classmethod
    def decode(cls, data, offset=0):
        if offset + 2 > len(data):
            return cls(0, 0)
        mode = data[offset]
        status = data[offset + 1]
        return cls(mode, status)

    def to_dict(self):
        return {
            'mode': self.mode,
            'mode_name': APP_INDEX_TO_NAME.get(self.mode, 'Unknown(%s)' % self.mode),
            'status': {
                'run': self.is_running,
                'result': self.has_result,
                'ready': self.is_ready,
                'busy': self.is_busy,
                'error': self.has_error,
            },
        }


# ============================================================================
# K230客户端基类
# ============================================================================
class K230ClientBase:
    """K230通信客户端基类

    包含共享的便捷命令方法和检测结果处理逻辑。
    子类需要实现: send_command, send_command_and_wait, _log
    """

    def __init__(self):
        self.parser = FrameParser()
        self.reassembler = PayloadReassembler()
        self.tx_seq = 0
        self.tx_txn = 1
        self.debug_mode = False
        self.result_validation_enabled = True
        self.current_heartbeat = HeartbeatData(-1)
        self._current_mode = 0
        self._current_mode_name = 'Empty'
        self._running = False
        self._last_validation_key = None
        self._last_validation_log_ms = 0
        self.last_detect_validation = None
        self.last_detect_raw_values = None
        self.last_detect_raw_text = None

        # 命令结果等待
        self._pending_cmd = None
        self._pending_txn = None
        self._cmd_result_data = None
        self._cmd_success = False
        self._last_response_txn = None
        self._rx_frame_serial = 0
        self._heartbeat_serial = 0
        self._report_serial = 0
        self._response_serial = 0
        self.last_rx_frame_ms = None
        self.last_rx_frame_type = None
        self.last_rx_frame_func_code = None
        self.last_rx_frame_payload_len = 0
        self.last_report_ms = None
        self.last_report_func_code = None
        self.last_report_kind = None
        self.last_report_payload_len = 0
        self.last_response_ms = None
        self.last_response_func_code = None
        self.last_response_payload_len = 0

        # 回调函数
        self.on_detect_result = None
        self.on_heartbeat = None
        self.on_command_result = None
        self.on_error = None
        # 异步结果回调
        self.on_llm_result = None
        self.on_vlm_result = None
        self.on_asr_result = None
        self.on_tts_finish = None
        self.on_wifi_connected = None
        self.on_mcp_result = None  # MCP工具结果回调
        # 异步结果等待
        self._async_result = None
        self._async_result_ready = False
        self._async_success = False
        self._waiting_async_cmd = None
        self._waiting_async_owner_cmd = None
        self._waiting_async_txn = None

    def _next_seq(self):
        """获取下一个发送序列号"""
        seq = self.tx_seq
        self.tx_seq = (self.tx_seq + 1) & SEQ_MASK
        return seq

    def _next_txn(self):
        """获取下一个事务号，0 保留给主动上报。"""
        txn = self.tx_txn & 0xFF
        if txn == 0:
            txn = 1
        self.tx_txn = ((txn + 1) & 0xFF) or 1
        return txn

    @property
    def current_mode(self):
        return self._current_mode

    @property
    def current_mode_name(self):
        return self._current_mode_name

    @property
    def rx_frame_serial(self):
        return self._rx_frame_serial

    @property
    def report_serial(self):
        return self._report_serial

    @property
    def response_serial(self):
        return self._response_serial

    @property
    def heartbeat_serial(self):
        return self._heartbeat_serial

    def get_heartbeat(self):
        """获取当前心跳数据"""
        return self.current_heartbeat.to_dict()

    def describe_last_frame(self):
        if self.last_rx_frame_func_code is None:
            return '--'
        return "%s/0x%02X/%dB" % (
            FRAME_TYPE_NAMES.get(self.last_rx_frame_type, "type_0x%02X" % self.last_rx_frame_type),
            self.last_rx_frame_func_code,
            self.last_rx_frame_payload_len,
        )

    def describe_last_report(self):
        if self.last_report_func_code is None:
            return '--'
        return "%s/0x%02X/%dB" % (
            self.last_report_kind or ("func_0x%02X" % self.last_report_func_code),
            self.last_report_func_code,
            self.last_report_payload_len,
        )

    def describe_last_response(self):
        if self.last_response_func_code is None:
            return '--'
        return "rsp/0x%02X/%dB" % (
            self.last_response_func_code,
            self.last_response_payload_len,
        )

    def has_recent_transport_activity(self, timeout_ms):
        if self.last_rx_frame_ms is None:
            return False
        return self._ticks_diff(self._ticks_ms(), self.last_rx_frame_ms) <= timeout_ms

    def _normalize_mode_index(self, mode):
        if isinstance(mode, str):
            mode_idx = APP_NAME_TO_INDEX.get(mode)
            if mode_idx is None:
                self._log("未知模式名称: %s" % mode)
                return None
            return mode_idx

        mode_idx = mode
        if mode_idx not in APP_INDEX_TO_NAME:
            self._log("未知模式索引: %s" % mode_idx)
            return None
        return mode_idx

    def _current_mode_index_for_switch(self):
        mode_idx = getattr(self.current_heartbeat, 'mode', -1)
        if mode_idx in APP_INDEX_TO_NAME:
            return mode_idx
        if self._current_mode in APP_INDEX_TO_NAME:
            return self._current_mode
        return None

    def _poll_transport_once(self):
        poll_status = getattr(self, 'poll_status', None)
        if callable(poll_status):
            try:
                poll_status()
            except Exception:
                return False
            return True
        return False

    def _heartbeat_matches(self, expected_mode=None, expected_run=None):
        heartbeat = getattr(self, 'current_heartbeat', None)
        if heartbeat is None:
            return False
        if expected_mode is not None and heartbeat.mode != expected_mode:
            return False
        if expected_run is not None and heartbeat.is_running != bool(expected_run):
            return False
        return True

    def wait_for_heartbeat_state(self,
                                 expected_mode=None,
                                 expected_run=None,
                                 timeout_ms=3000,
                                 min_heartbeat_serial=None,
                                 request_status_after_ms=120):
        start = self._ticks_ms()
        last_status_request_ms = None
        request_status = getattr(self, 'request_status', None)
        request_status_used = False

        while self._ticks_diff(self._ticks_ms(), start) < timeout_ms:
            if self._heartbeat_matches(expected_mode, expected_run):
                if min_heartbeat_serial is None or self._heartbeat_serial >= min_heartbeat_serial:
                    return True

            self._poll_transport_once()

            now = self._ticks_ms()
            if (
                callable(request_status) and
                request_status_after_ms is not None and
                self._ticks_diff(now, start) >= request_status_after_ms and
                (not request_status_used or
                 (last_status_request_ms is not None and
                  self._ticks_diff(now, last_status_request_ms) >= 250))
            ):
                remaining_ms = timeout_ms - self._ticks_diff(now, start)
                if remaining_ms <= 0:
                    break
                try:
                    wait_budget_ms = min(remaining_ms, 600)
                    request_status(timeout_ms=wait_budget_ms)
                except TypeError:
                    request_status()
                except Exception:
                    pass
                request_status_used = True
                last_status_request_ms = self._ticks_ms()
                continue

            self._sleep_ms(5)

        return False

    def _set_mode_and_confirm_mode(self, mode_idx, timeout_ms=3000):
        prev_mode_idx = self._current_mode_index_for_switch()
        start_heartbeat_serial = self._heartbeat_serial
        payload = CompactCodec.encode_uint8(mode_idx)
        success, _ = self.send_command_and_wait(CMD_SET_MODE, payload, timeout_ms=timeout_ms)
        if not success:
            return False

        # Current heartbeat does not expose the MCU run_enabled state. Only wait
        # for a fresh heartbeat when the externally visible mode is expected to
        # change; same-mode reset/re-entry must rely on the synchronous response.
        if prev_mode_idx != mode_idx:
            if not self.wait_for_heartbeat_state(expected_mode=mode_idx,
                                                 timeout_ms=timeout_ms,
                                                 min_heartbeat_serial=start_heartbeat_serial + 1):
                self._log("等待模式切换心跳超时: mode=%s" %
                          APP_INDEX_TO_NAME.get(mode_idx, 'Unknown(%s)' % mode_idx))
                return False

        self._current_mode = mode_idx
        self._current_mode_name = APP_INDEX_TO_NAME.get(mode_idx, 'Unknown(%s)' % mode_idx)
        return True

    # ========================================================================
    # 子类需要实现的方法
    # ========================================================================

    def send_command(self, func_code, payload=b''):
        """发送命令 - 子类实现"""
        raise NotImplementedError

    def send_command_and_wait(self, func_code, payload=b'', timeout_ms=3000):
        """发送命令并等待结果 - 子类实现"""
        raise NotImplementedError

    def send_command_and_wait_async_result(self, func_code, payload=b'', timeout_ms=30000, wait_cmd=None):
        """发送命令并等待异步结果

        用于 LLM/VLM/ASR/TTS 等需要等待异步结果的命令。
        流程：发送命令 -> 等待命令确认 -> 继续等待异步结果

        Args:
            func_code: 命令功能码
            payload: 命令载荷
            timeout_ms: 超时时间（默认30秒，LLM可能需要较长时间）
            wait_cmd: 等待的结果命令码，默认为 CMD_RESULT_RETURN

        Returns:
            (success, result): 成功标志和结果数据
        """
        if wait_cmd is None:
            wait_cmd = CMD_RESULT_RETURN

        # 清空之前的异步结果
        self._async_result = None
        self._async_result_ready = False
        self._async_success = False
        self._waiting_async_cmd = wait_cmd
        self._waiting_async_owner_cmd = func_code
        self._waiting_async_txn = None

        # 发送命令并等待确认
        success, _ = self.send_command_and_wait(func_code, payload, timeout_ms=5000)
        if not success:
            self._waiting_async_cmd = None
            self._waiting_async_owner_cmd = None
            self._waiting_async_txn = None
            return False, None
        self._waiting_async_txn = self._last_response_txn

        # 继续等待异步结果 - 子类需要在轮询中调用 _check_async_result
        # 这里提供默认实现，子类可以覆盖
        start = self._ticks_ms()
        while self._ticks_diff(self._ticks_ms(), start) < timeout_ms:
            if self._async_result_ready:
                result = self._async_result
                success = self._async_success
                self._async_result = None
                self._async_result_ready = False
                self._async_success = False
                self._waiting_async_cmd = None
                self._waiting_async_owner_cmd = None
                self._waiting_async_txn = None
                return success, result
            self._sleep_ms(10)

        self._waiting_async_cmd = None
        self._waiting_async_owner_cmd = None
        self._waiting_async_txn = None
        return False, None

    def _log(self, msg):
        """日志输出 - 子类实现"""
        raise NotImplementedError

    def _ticks_ms(self):
        try:
            return time.ticks_ms()
        except AttributeError:
            return int(time.time() * 1000)

    def _ticks_diff(self, new_ticks, old_ticks):
        try:
            return time.ticks_diff(new_ticks, old_ticks)
        except AttributeError:
            return new_ticks - old_ticks

    def _sleep_ms(self, duration_ms):
        try:
            time.sleep_ms(duration_ms)
        except AttributeError:
            time.sleep(duration_ms / 1000.0)

    def _note_rx_frame_activity(self, frame_type, func_code, payload_len):
        now = self._ticks_ms()
        self._rx_frame_serial += 1
        self.last_rx_frame_ms = now
        self.last_rx_frame_type = frame_type
        self.last_rx_frame_func_code = func_code
        self.last_rx_frame_payload_len = payload_len

    def _note_report_activity(self, func_code, payload_len):
        self._report_serial += 1
        self.last_report_ms = self._ticks_ms()
        self.last_report_func_code = func_code
        self.last_report_kind = REPORT_KIND_NAMES.get(func_code, "func_0x%02X" % func_code)
        self.last_report_payload_len = payload_len

    def _note_response_activity(self, func_code, payload_len):
        self._response_serial += 1
        self.last_response_ms = self._ticks_ms()
        self.last_response_func_code = func_code
        self.last_response_payload_len = payload_len

    def _decode_prefixed_text_lossy(self, data, offset):
        """Best-effort decode for length-prefixed OCR/text payloads.

        Returns:
            (text, consumed, truncated, replaced, declared_len, actual_len)
        """
        if offset >= len(data):
            return "", 0, True, False, 0, 0

        declared_len = data[offset]
        start = offset + 1
        end = start + declared_len
        actual_end = min(end, len(data))
        raw = data[start:actual_end]
        consumed = 1 + len(raw)
        truncated = actual_end != end
        replaced = False

        try:
            text = raw.decode('utf-8')
        except UnicodeError:
            replaced = True
            text = raw.decode('utf-8', 'replace')

        return text, consumed, truncated, replaced, declared_len, len(raw)

    def _decode_raw_string_values(self, data, offset):
        if offset >= len(data):
            return "", 0, 0
        declared_len = data[offset]
        text, consumed = CompactCodec.decode_string(data, offset)
        return text, consumed, declared_len

    def _format_raw_scalar(self, value):
        if isinstance(value, float):
            return ("%0.2f" % value).rstrip('0').rstrip('.')
        return str(value)

    def _join_labeled_raw_values(self, raw_values, labels):
        parts = []
        for idx, value in enumerate(raw_values):
            label = labels[idx] if idx < len(labels) else ""
            value_text = self._format_raw_scalar(value)
            if label:
                parts.append("%s(%s)" % (value_text, label))
            else:
                parts.append(value_text)
        return "[" + ", ".join(parts) + "]"

    def _build_bbox_extra_labels(self, extras):
        labels = ['extra_count']
        for extra in extras:
            if isinstance(extra, str):
                labels.extend(('extra_type=string', 'string_len', 'string_value'))
            elif isinstance(extra, float):
                labels.extend(('extra_type=float_x100', 'float_value'))
            elif isinstance(extra, list):
                labels.extend(('extra_type=int_list', 'list_len'))
                labels.extend('list_value' for _ in extra)
            else:
                labels.extend(('extra_type=int', 'int_value'))
        return labels

    def _build_center_extra_labels(self, extras):
        labels = ['extra_count']
        for extra in extras:
            if isinstance(extra, str):
                labels.extend(('extra_type=string', 'string_len', 'string_value'))
            elif isinstance(extra, float):
                labels.extend(('extra_type=float_x100', 'float_value'))
            else:
                labels.extend(('extra_type=int', 'int_value'))
        return labels

    def _build_raw_value_labels(self, parsed, raw_values):
        if not isinstance(parsed, dict) or not isinstance(raw_values, list):
            return []

        result_type = parsed.get('type')
        labels = []

        if result_type == 'bbox':
            labels.append('count')
            for item in parsed.get('results', []):
                labels.extend(('x', 'y', 'w', 'h'))
                labels.extend(self._build_bbox_extra_labels(item.get('extra', [])))
            return labels

        if result_type == 'quad':
            labels.append('count')
            for item in parsed.get('results', []):
                point_count = len(item.get('points', [])) // 2
                for idx in range(point_count):
                    labels.extend(("x%d" % idx, "y%d" % idx))
                labels.extend(self._build_bbox_extra_labels(item.get('extra', [])))
            return labels

        if result_type == 'string':
            return ['value_len', 'value']

        if result_type == 'string_list':
            labels.append('count')
            for _ in parsed.get('results', []):
                labels.extend(('value_len', 'value'))
            return labels

        if result_type == 'ocr':
            labels.extend(('count', 'has_text'))
            has_text = bool(len(raw_values) > 1 and raw_values[1])
            for _ in parsed.get('results', []):
                for idx in range(4):
                    labels.extend(("x%d" % idx, "y%d" % idx))
                if has_text:
                    labels.extend(('text_len', 'text'))
            return labels

        if result_type in ('color', 'multi_color'):
            if result_type == 'color':
                blobs = parsed.get('blobs', [])
                rotated = any(blob.get('geometry') == 'rotated_rect' for blob in blobs)
                if raw_values and raw_values[0] in (0, 2):
                    labels.append('single_color_rotated' if raw_values[0] == 2 else 'single_color')
                    labels.extend(('name_len', 'name', 'num'))
                else:
                    labels.extend(('num', 'name_len', 'name'))
                for _ in blobs:
                    labels.extend(('cx', 'cy', 'w', 'h', 'angle') if rotated else ('x', 'y', 'w', 'h', 'angle'))
                return labels

            groups = parsed.get('results', [])
            rotated = any(
                blob.get('geometry') == 'rotated_rect'
                for group in groups
                for blob in group.get('blobs', [])
            )
            labels.append('num')
            for group in groups:
                labels.extend(('name_len', 'name', 'num'))
                for _ in group.get('blobs', []):
                    labels.extend(('cx', 'cy', 'w', 'h', 'angle') if rotated else ('x', 'y', 'w', 'h', 'angle'))
            return labels

        if result_type == 'line':
            labels.extend(('num', 'name_len', 'name', 'center_pos', 'angle'))
            for _ in parsed.get('blobs', []):
                labels.extend(('index', 'x', 'y', 'w', 'h'))
            return labels

        if result_type == 'keypoint':
            labels.append('count')
            for item in parsed.get('results', []):
                labels.extend("kp%d" % idx for idx in range(34))
                labels.extend(('id_len', 'id'))
                if 'score' in item:
                    labels.append('score')
            return labels

        if result_type == 'hand_kp':
            labels.append('count')
            for item in parsed.get('results', []):
                labels.extend(('x', 'y', 'w', 'h'))
                labels.extend("kp%d" % idx for idx in range(42))
                labels.extend(('id_len', 'id'))
                if 'score' in item:
                    labels.append('score')
            return labels

        if result_type == 'center':
            labels.append('count')
            for item in parsed.get('results', []):
                labels.extend(('x', 'y'))
                labels.extend(self._build_center_extra_labels(item.get('extra', [])))
            return labels

        if result_type == 'face_kp':
            labels.append('count')
            for _ in parsed.get('results', []):
                labels.extend("kp%d" % idx for idx in range(10))
            return labels

        return labels

    def _format_detect_raw_values(self, parsed, raw_values):
        if not isinstance(raw_values, list):
            return None
        labels = self._build_raw_value_labels(parsed, raw_values)
        return self._join_labeled_raw_values(raw_values, labels)

    def _add_validation_issue(self, issues, message):
        if len(issues) < MAX_RESULT_VALIDATION_ISSUES:
            issues.append(message)

    def _is_int_like(self, value):
        return isinstance(value, int) and not isinstance(value, bool)

    def _is_number(self, value):
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)

    def _validate_string_field(self, value, path, issues, allow_empty=True):
        if not isinstance(value, str):
            self._add_validation_issue(issues, "%s should be string" % path)
            return
        if not allow_empty and value == '':
            self._add_validation_issue(issues, "%s should not be empty" % path)

    def _validate_enum_string(self, value, path, issues, allowed_values, allow_empty=False):
        self._validate_string_field(value, path, issues, allow_empty=allow_empty)
        if not isinstance(value, str):
            return
        if allow_empty and value == '':
            return
        if value not in allowed_values:
            self._add_validation_issue(issues, "%s=%r not in allowed set" % (path, value))

    def _validate_int_range(self, value, path, issues, min_value=None, max_value=None):
        if not self._is_int_like(value):
            self._add_validation_issue(issues, "%s should be int" % path)
            return
        if min_value is not None and value < min_value:
            self._add_validation_issue(issues, "%s=%s < %s" % (path, value, min_value))
        if max_value is not None and value > max_value:
            self._add_validation_issue(issues, "%s=%s > %s" % (path, value, max_value))

    def _validate_point(self, x, y, path, issues, allow_zero_pair=False,
                        min_x=0, max_x=RESULT_COORD_MAX_X,
                        min_y=0, max_y=RESULT_COORD_MAX_Y):
        if not self._is_int_like(x) or not self._is_int_like(y):
            self._add_validation_issue(issues, "%s should be int point pair" % path)
            return
        if allow_zero_pair and x == 0 and y == 0:
            return
        if x < min_x or x > max_x:
            self._add_validation_issue(issues, "%s.x=%s out of range" % (path, x))
        if y < min_y or y > max_y:
            self._add_validation_issue(issues, "%s.y=%s out of range" % (path, y))

    def _validate_bbox_geometry(self, item, path, issues):
        x = item.get('x')
        y = item.get('y')
        w = item.get('w')
        h = item.get('h')
        self._validate_int_range(x, path + '.x', issues, 0, RESULT_COORD_MAX_X)
        self._validate_int_range(y, path + '.y', issues, 0, RESULT_COORD_MAX_Y)
        self._validate_int_range(w, path + '.w', issues, 1, RESULT_COORD_MAX_X)
        self._validate_int_range(h, path + '.h', issues, 1, RESULT_COORD_MAX_Y)
        if self._is_int_like(x) and self._is_int_like(w) and (x + w) > RESULT_COORD_MAX_X:
            self._add_validation_issue(issues, "%s.x+w=%s out of range" % (path, x + w))
        if self._is_int_like(y) and self._is_int_like(h) and (y + h) > RESULT_COORD_MAX_Y:
            self._add_validation_issue(issues, "%s.y+h=%s out of range" % (path, y + h))

    def _validate_rotated_rect_geometry(self, item, path, issues):
        cx = item.get('cx')
        cy = item.get('cy')
        w = item.get('w')
        h = item.get('h')
        self._validate_int_range(cx, path + '.cx', issues, 0, RESULT_COORD_MAX_X)
        self._validate_int_range(cy, path + '.cy', issues, 0, RESULT_COORD_MAX_Y)
        self._validate_int_range(w, path + '.w', issues, 1, RESULT_COORD_MAX_X)
        self._validate_int_range(h, path + '.h', issues, 1, RESULT_COORD_MAX_Y)

    def _validate_point_list(self, values, expected_len, path, issues, allow_zero_pair=False,
                             min_x=0, max_x=RESULT_COORD_MAX_X,
                             min_y=0, max_y=RESULT_COORD_MAX_Y):
        if not isinstance(values, list):
            self._add_validation_issue(issues, "%s should be list" % path)
            return
        if len(values) != expected_len:
            self._add_validation_issue(issues, "%s len=%s expected=%s" % (path, len(values), expected_len))
            return
        idx = 0
        while idx + 1 < len(values):
            self._validate_point(values[idx], values[idx + 1],
                                 "%s[%d:%d]" % (path, idx, idx + 1),
                                 issues,
                                 allow_zero_pair=allow_zero_pair,
                                 min_x=min_x,
                                 max_x=max_x,
                                 min_y=min_y,
                                 max_y=max_y)
            idx += 2

    def _validate_bbox_extras(self, mode_name, extras, path, issues):
        if not isinstance(extras, list):
            self._add_validation_issue(issues, "%s should be list" % path)
            return

        if mode_name in ('FaceRecognition', 'FaceLiveness'):
            if len(extras) == 0:
                return
            if len(extras) == 2:
                self._validate_string_field(extras[0], path + '[0]', issues, allow_empty=False)
                self._validate_int_range(extras[1], path + '[1]', issues, 0, RESULT_SCORE_MAX)
                return
            if mode_name == 'FaceRecognition' and len(extras) == 10:
                self._validate_point_list(extras, 10, path, issues)
                return
            self._add_validation_issue(issues, "%s len=%s unexpected for %s" %
                                       (path, len(extras), mode_name))
            return

        if mode_name in ('FaceDetection', 'PersonDetection', 'ObjectTrack',
                         'HandDetection', 'LicencePlateDetection'):
            if len(extras) != 0:
                self._add_validation_issue(issues, "%s len=%s expected=0" % (path, len(extras)))
            return

        if mode_name in ('FaceLandmark', 'FaceMesh'):
            if len(extras) == 0:
                return
            if len(extras) == 10:
                self._validate_point_list(extras, 10, path, issues)
                return
            self._add_validation_issue(issues, "%s len=%s unexpected for %s" %
                                       (path, len(extras), mode_name))
            return

        if mode_name == 'FacePose':
            pose_values = extras
            if pose_values and isinstance(pose_values[0], str):
                self._validate_string_field(pose_values[0], path + '[0]', issues, allow_empty=False)
                pose_values = pose_values[1:]
            if len(pose_values) != 3:
                self._add_validation_issue(issues, "%s len=%s expected=3/4" % (path, len(extras)))
                return
            for idx, value in enumerate(pose_values):
                self._validate_int_range(value, "%s[%d]" % (path, idx), issues, -180, 180)
            return

        if mode_name == 'EyeGaze':
            gaze_values = extras
            if gaze_values and isinstance(gaze_values[0], str):
                self._validate_string_field(gaze_values[0], path + '[0]', issues, allow_empty=False)
                gaze_values = gaze_values[1:]
            if len(gaze_values) != 4:
                self._add_validation_issue(issues, "%s len=%s expected=4/5" % (path, len(extras)))
                return
            self._validate_point(gaze_values[0], gaze_values[1], path + '[0:1]', issues)
            self._validate_point(gaze_values[2], gaze_values[3],
                                 path + '[2:3]',
                                 issues,
                                 min_x=RESULT_EYE_GAZE_TARGET_MIN_X,
                                 max_x=RESULT_EYE_GAZE_TARGET_MAX_X,
                                 min_y=RESULT_EYE_GAZE_TARGET_MIN_Y,
                                 max_y=RESULT_EYE_GAZE_TARGET_MAX_Y)
            return

        if mode_name in ('HandGesture', 'HandRecognition'):
            if len(extras) not in (0, 1):
                self._add_validation_issue(issues, "%s len=%s unexpected" % (path, len(extras)))
                return
            if len(extras) == 1:
                self._validate_enum_string(extras[0], path + '[0]', issues, HAND_GESTURE_LABELS)
            return

        if mode_name == 'FaceParse':
            if len(extras) != 0:
                self._add_validation_issue(issues, "%s len=%s expected=0" % (path, len(extras)))
            return

        if mode_name == 'SelfLearning':
            if len(extras) != 2:
                self._add_validation_issue(issues, "%s len=%s expected=2" % (path, len(extras)))
                return
            self._validate_string_field(extras[0], path + '[0]', issues, allow_empty=False)
            self._validate_int_range(extras[1], path + '[1]', issues, 0, RESULT_SCORE_MAX)
            return

        if mode_name == 'FalldownDetection':
            if len(extras) != 2:
                self._add_validation_issue(issues, "%s len=%s expected=2" % (path, len(extras)))
                return
            self._validate_int_range(extras[0], path + '[0]', issues, 0, 1)
            self._validate_int_range(extras[1], path + '[1]', issues, 0, RESULT_SCORE_MAX)
            return

        if mode_name == 'GarbageClassification':
            if len(extras) != 2:
                self._add_validation_issue(issues, "%s len=%s expected=2" % (path, len(extras)))
                return
            self._validate_enum_string(extras[0], path + '[0]', issues, GARBAGE_CLASS_LABELS)
            self._validate_int_range(extras[1], path + '[1]', issues, 0, RESULT_SCORE_MAX)
            return

        if mode_name == 'TrafficDetection':
            if len(extras) != 2:
                self._add_validation_issue(issues, "%s len=%s expected=2" % (path, len(extras)))
                return
            self._validate_enum_string(extras[0], path + '[0]', issues, TRAFFIC_LABELS)
            self._validate_int_range(extras[1], path + '[1]', issues, 0, RESULT_SCORE_MAX)
            return

        if mode_name in ('ObjectDetection', 'Segmentation', 'CustomDetection'):
            if len(extras) == 1:
                self._validate_int_range(extras[0], path + '[0]', issues, 0, RESULT_SCORE_MAX)
                return
            if len(extras) == 2:
                self._validate_string_field(extras[0], path + '[0]', issues, allow_empty=False)
                self._validate_int_range(extras[1], path + '[1]', issues, 0, RESULT_SCORE_MAX)
                return
            self._add_validation_issue(issues, "%s len=%s unexpected for detection bbox" % (path, len(extras)))
            return

        if mode_name == 'ApriltagDiscern':
            if len(extras) not in (1, 2):
                self._add_validation_issue(issues, "%s len=%s unexpected for ApriltagDiscern" % (path, len(extras)))
                return
            self._validate_string_field(extras[0], path + '[0]', issues, allow_empty=False)
            if len(extras) == 2:
                self._validate_int_range(extras[1], path + '[1]', issues, 0, None)
            return

        if mode_name in ('DMCodeDiscern', 'QRCodeDiscern', 'BarCodeDiscern'):
            if len(extras) not in (0, 1):
                self._add_validation_issue(issues, "%s len=%s unexpected for code bbox" % (path, len(extras)))
                return
            if len(extras) == 1:
                self._validate_string_field(extras[0], path + '[0]', issues, allow_empty=False)
            return

        if mode_name in ('HandKeyPointDetection',):
            if len(extras) != 0:
                self._add_validation_issue(issues, "%s len=%s expected=0" % (path, len(extras)))
            return

    def _validate_center_extras(self, mode_name, extras, path, issues):
        if not isinstance(extras, list):
            self._add_validation_issue(issues, "%s should be list" % path)
            return

        if mode_name in ('FaceRecognition', 'FaceLiveness'):
            if len(extras) not in (0, 1):
                self._add_validation_issue(issues, "%s len=%s unexpected for %s" %
                                           (path, len(extras), mode_name))
            if len(extras) == 1:
                self._validate_string_field(extras[0], path + '[0]', issues, allow_empty=False)
            return

        if mode_name == 'FacePose':
            pose_values = extras
            if pose_values and isinstance(pose_values[0], str):
                self._validate_string_field(pose_values[0], path + '[0]', issues, allow_empty=False)
                pose_values = pose_values[1:]
            if len(pose_values) != 3:
                self._add_validation_issue(issues, "%s len=%s expected=3/4" % (path, len(extras)))
                return
            for idx, value in enumerate(pose_values):
                self._validate_int_range(value, "%s[%d]" % (path, idx), issues, -180, 180)
            return

        if mode_name == 'EyeGaze':
            gaze_values = extras
            if gaze_values and isinstance(gaze_values[0], str):
                self._validate_string_field(gaze_values[0], path + '[0]', issues, allow_empty=False)
                gaze_values = gaze_values[1:]
            if len(gaze_values) != 2:
                self._add_validation_issue(issues, "%s len=%s expected=2/3" % (path, len(extras)))
                return
            self._validate_point(gaze_values[0], gaze_values[1],
                                 path,
                                 issues,
                                 min_x=RESULT_EYE_GAZE_TARGET_MIN_X,
                                 max_x=RESULT_EYE_GAZE_TARGET_MAX_X,
                                 min_y=RESULT_EYE_GAZE_TARGET_MIN_Y,
                                 max_y=RESULT_EYE_GAZE_TARGET_MAX_Y)
            return

        if mode_name in ('HandGesture', 'HandRecognition'):
            if len(extras) != 1:
                self._add_validation_issue(issues, "%s len=%s expected=1" % (path, len(extras)))
                return
            self._validate_enum_string(extras[0], path + '[0]', issues, HAND_GESTURE_LABELS)
            return

        if mode_name == 'FaceParse':
            if len(extras) != 0:
                self._add_validation_issue(issues, "%s len=%s expected=0" % (path, len(extras)))
            return

        if mode_name in ('SingleColorDetection', 'MultiColorDetection',
                         'ColorTracking', 'OCRRecognition', 'LicencePlateRecognition',
                         'ObjectDetection', 'Segmentation', 'CustomDetection',
                         'DMCodeDiscern'):
            if len(extras) not in (0, 1):
                self._add_validation_issue(issues, "%s len=%s unexpected" % (path, len(extras)))
                return
            if len(extras) == 1:
                self._validate_string_field(extras[0], path + '[0]', issues, allow_empty=False)
            return

        if mode_name in ('FaceDetection', 'FaceLandmark', 'FaceMesh',
                         'PersonDetection', 'ObjectTrack', 'HandDetection',
                         'HandKeyPointDetection', 'OCRDetection', 'LicencePlateDetection'):
            if len(extras) != 0:
                self._add_validation_issue(issues, "%s len=%s expected=0" % (path, len(extras)))
            return

        if mode_name == 'GarbageClassification':
            if len(extras) != 1:
                self._add_validation_issue(issues, "%s len=%s expected=1" % (path, len(extras)))
                return
            self._validate_enum_string(extras[0], path + '[0]', issues, GARBAGE_CLASS_LABELS)
            return

        if mode_name == 'TrafficDetection':
            if len(extras) != 1:
                self._add_validation_issue(issues, "%s len=%s expected=1" % (path, len(extras)))
                return
            self._validate_enum_string(extras[0], path + '[0]', issues, TRAFFIC_LABELS)
            return

        if mode_name == 'ApriltagDiscern':
            if len(extras) not in (1, 2):
                self._add_validation_issue(issues, "%s len=%s unexpected for ApriltagDiscern" % (path, len(extras)))
                return
            self._validate_string_field(extras[0], path + '[0]', issues, allow_empty=False)
            if len(extras) == 2:
                self._validate_int_range(extras[1], path + '[1]', issues, 0, None)
            return

        if mode_name == 'LineDetection':
            if len(extras) != 1:
                self._add_validation_issue(issues, "%s len=%s expected=1" % (path, len(extras)))
                return
            self._validate_int_range(extras[0], path + '[0]', issues,
                                     -RESULT_ANGLE_ABS_MAX, RESULT_ANGLE_ABS_MAX)
            return

        if mode_name == 'FalldownDetection':
            if len(extras) != 2:
                self._add_validation_issue(issues, "%s len=%s expected=2" % (path, len(extras)))
                return
            self._validate_int_range(extras[0], path + '[0]', issues, 0, 1)
            self._validate_int_range(extras[1], path + '[1]', issues, 0, RESULT_SCORE_MAX)
            return

    def _validate_bbox_result(self, mode_name, parsed, issues):
        results = parsed.get('results')
        if not isinstance(results, list):
            self._add_validation_issue(issues, "results should be list")
            return
        for idx, item in enumerate(results):
            path = "results[%d]" % idx
            if not isinstance(item, dict):
                self._add_validation_issue(issues, "%s should be dict" % path)
                continue
            self._validate_bbox_geometry(item, path, issues)
            extras = item.get('extra', [])
            self._validate_bbox_extras(mode_name, extras, path + '.extra', issues)

    def _validate_quad_result(self, mode_name, parsed, issues):
        results = parsed.get('results')
        if not isinstance(results, list):
            self._add_validation_issue(issues, "results should be list")
            return
        for idx, item in enumerate(results):
            path = "results[%d]" % idx
            if not isinstance(item, dict):
                self._add_validation_issue(issues, "%s should be dict" % path)
                continue
            self._validate_point_list(item.get('points'),
                                      8,
                                      path + '.points',
                                      issues,
                                      min_x=0,
                                      max_x=RESULT_COORD_MAX_X,
                                      min_y=0,
                                      max_y=RESULT_COORD_MAX_Y)
            extras = item.get('extra', [])
            self._validate_bbox_extras(mode_name, extras, path + '.extra', issues)

    def _validate_center_result(self, mode_name, parsed, issues):
        results = parsed.get('results')
        if not isinstance(results, list):
            self._add_validation_issue(issues, "results should be list")
            return
        for idx, item in enumerate(results):
            path = "results[%d]" % idx
            if not isinstance(item, dict):
                self._add_validation_issue(issues, "%s should be dict" % path)
                continue
            self._validate_point(item.get('x'), item.get('y'), path, issues)
            extras = item.get('extra', [])
            self._validate_center_extras(mode_name, extras, path + '.extra', issues)

    def _validate_string_result(self, parsed, issues):
        self._validate_string_field(parsed.get('value'), 'value', issues, allow_empty=False)

    def _validate_string_list_result(self, parsed, issues):
        results = parsed.get('results')
        if not isinstance(results, list):
            self._add_validation_issue(issues, "results should be list")
            return
        for idx, value in enumerate(results):
            self._validate_string_field(value, "results[%d]" % idx, issues, allow_empty=False)

    def _validate_ocr_result(self, mode_name, parsed, issues):
        results = parsed.get('results')
        if not isinstance(results, list):
            self._add_validation_issue(issues, "results should be list")
            return
        min_x = 0
        max_x = RESULT_COORD_MAX_X
        min_y = 0
        max_y = RESULT_COORD_MAX_Y

        if mode_name == 'LicencePlateRecognition':
            min_x = RESULT_OCR_MIN_X
            max_x = RESULT_OCR_MAX_X
            min_y = RESULT_OCR_MIN_Y
            max_y = RESULT_OCR_MAX_Y

        for idx, item in enumerate(results):
            path = "results[%d]" % idx
            if not isinstance(item, dict):
                self._add_validation_issue(issues, "%s should be dict" % path)
                continue
            self._validate_point_list(item.get('points'),
                                      8,
                                      path + '.points',
                                      issues,
                                      min_x=min_x,
                                      max_x=max_x,
                                      min_y=min_y,
                                      max_y=max_y)
            self._validate_string_field(item.get('text', ''), path + '.text', issues)

    def _validate_color_blob(self, blob, path, issues):
        if not isinstance(blob, dict):
            self._add_validation_issue(issues, "%s should be dict" % path)
            return
        if 'cx' in blob or 'cy' in blob:
            self._validate_rotated_rect_geometry(blob, path, issues)
        else:
            self._validate_bbox_geometry(blob, path, issues)
        self._validate_int_range(blob.get('angle'), path + '.angle', issues,
                                 -RESULT_ANGLE_ABS_MAX, RESULT_ANGLE_ABS_MAX)

    def _validate_color_result(self, parsed, issues):
        self._validate_string_field(parsed.get('color'), 'color', issues, allow_empty=False)
        blobs = parsed.get('blobs')
        if not isinstance(blobs, list):
            self._add_validation_issue(issues, "blobs should be list")
            return
        for idx, blob in enumerate(blobs):
            self._validate_color_blob(blob, "blobs[%d]" % idx, issues)

    def _validate_multi_color_result(self, parsed, issues):
        results = parsed.get('results')
        if not isinstance(results, list):
            self._add_validation_issue(issues, "results should be list")
            return
        for idx, group in enumerate(results):
            path = "results[%d]" % idx
            if not isinstance(group, dict):
                self._add_validation_issue(issues, "%s should be dict" % path)
                continue
            self._validate_string_field(group.get('color'), path + '.color', issues, allow_empty=False)
            blobs = group.get('blobs')
            if not isinstance(blobs, list):
                self._add_validation_issue(issues, "%s.blobs should be list" % path)
                continue
            for blob_idx, blob in enumerate(blobs):
                self._validate_color_blob(blob, "%s.blobs[%d]" % (path, blob_idx), issues)

    def _validate_line_result(self, parsed, issues):
        self._validate_string_field(parsed.get('color'), 'color', issues, allow_empty=False)
        self._validate_int_range(parsed.get('center_pos'), 'center_pos', issues, 0, RESULT_COORD_MAX_X)
        self._validate_int_range(parsed.get('angle'), 'angle', issues,
                                 -RESULT_ANGLE_ABS_MAX, RESULT_ANGLE_ABS_MAX)
        blobs = parsed.get('blobs')
        if not isinstance(blobs, list):
            self._add_validation_issue(issues, "blobs should be list")
            return
        for idx, blob in enumerate(blobs):
            path = "blobs[%d]" % idx
            if not isinstance(blob, dict):
                self._add_validation_issue(issues, "%s should be dict" % path)
                continue
            self._validate_int_range(blob.get('index'),
                                     path + '.index',
                                     issues,
                                     0,
                                     RESULT_LINE_ROI_COUNT - 1)
            self._validate_bbox_geometry(blob, path, issues)

    def _validate_keypoint_result(self, parsed, issues):
        results = parsed.get('results')
        if not isinstance(results, list):
            self._add_validation_issue(issues, "results should be list")
            return
        for idx, item in enumerate(results):
            path = "results[%d]" % idx
            if not isinstance(item, dict):
                self._add_validation_issue(issues, "%s should be dict" % path)
                continue
            self._validate_point_list(item.get('keypoints'),
                                      34,
                                      path + '.keypoints',
                                      issues,
                                      allow_zero_pair=True,
                                      min_x=RESULT_KEYPOINT_MIN_X,
                                      max_x=RESULT_KEYPOINT_MAX_X,
                                      min_y=RESULT_KEYPOINT_MIN_Y,
                                      max_y=RESULT_KEYPOINT_MAX_Y)
            if 'id' in item:
                self._validate_string_field(item.get('id'), path + '.id', issues, allow_empty=True)
            if 'name' in item:
                self._validate_string_field(item.get('name'), path + '.name', issues, allow_empty=True)
            if 'score' in item:
                self._validate_int_range(item.get('score'), path + '.score', issues, 0, RESULT_SCORE_MAX)
            if 'name_len' in item:
                self._validate_int_range(item.get('name_len'), path + '.name_len', issues, 0, 255)

    def _validate_hand_kp_result(self, parsed, issues):
        results = parsed.get('results')
        if not isinstance(results, list):
            self._add_validation_issue(issues, "results should be list")
            return
        for idx, item in enumerate(results):
            path = "results[%d]" % idx
            if not isinstance(item, dict):
                self._add_validation_issue(issues, "%s should be dict" % path)
                continue
            self._validate_bbox_geometry(item, path, issues)
            self._validate_point_list(item.get('keypoints'),
                                      42,
                                      path + '.keypoints',
                                      issues,
                                      allow_zero_pair=True,
                                      min_x=RESULT_KEYPOINT_MIN_X,
                                      max_x=RESULT_KEYPOINT_MAX_X,
                                      min_y=RESULT_KEYPOINT_MIN_Y,
                                      max_y=RESULT_KEYPOINT_MAX_Y)
            if 'id' in item:
                self._validate_string_field(item.get('id'), path + '.id', issues, allow_empty=True)
            if 'name' in item:
                self._validate_string_field(item.get('name'), path + '.name', issues, allow_empty=True)
            if 'score' in item:
                self._validate_int_range(item.get('score'), path + '.score', issues, 0, RESULT_SCORE_MAX)
            if 'name_len' in item:
                self._validate_int_range(item.get('name_len'), path + '.name_len', issues, 0, 255)

    def _validate_face_kp_result(self, parsed, issues):
        results = parsed.get('results')
        if not isinstance(results, list):
            self._add_validation_issue(issues, "results should be list")
            return
        for idx, item in enumerate(results):
            path = "results[%d]" % idx
            if not isinstance(item, dict):
                self._add_validation_issue(issues, "%s should be dict" % path)
                continue
            self._validate_point_list(item.get('keypoints'),
                                      10,
                                      path + '.keypoints',
                                      issues,
                                      min_x=RESULT_KEYPOINT_MIN_X,
                                      max_x=RESULT_KEYPOINT_MAX_X,
                                      min_y=RESULT_KEYPOINT_MIN_Y,
                                      max_y=RESULT_KEYPOINT_MAX_Y)

    def _log_validation_issues(self, mode_name, result_type, issues):
        if not issues:
            return
        summary = '; '.join(issues)
        key = "%s|%s|%s" % (mode_name, result_type, summary)
        now = self._ticks_ms()
        if key == self._last_validation_key and self._ticks_diff(now, self._last_validation_log_ms) < 1000:
            return
        self._last_validation_key = key
        self._last_validation_log_ms = now
        result = None
        if isinstance(self.last_detect_validation, dict):
            result = self.last_detect_validation.get('result')
        self._log("[VALIDATE] mode=%s type=%s data=%s invalid: %s" %
                  (mode_name, result_type, result, summary))

    def _is_empty_bbox_fallback(self, parsed):
        """MicroPython compatibility: empty list results are encoded as empty bbox frames."""
        if not isinstance(parsed, dict):
            return False
        if parsed.get('type') != 'bbox':
            return False
        results = parsed.get('results')
        return isinstance(results, list) and len(results) == 0

    def _validate_detect_result(self, parsed):
        if not self.result_validation_enabled or not isinstance(parsed, dict):
            self.last_detect_validation = None
            return

        mode_name = self.current_mode_name
        result_type = parsed.get('type')
        issues = []
        allowed_types = MODE_ALLOWED_RESULT_TYPES.get(mode_name)
        empty_bbox_fallback = self._is_empty_bbox_fallback(parsed)

        if not isinstance(result_type, str):
            self._add_validation_issue(issues, "type should be string")
        elif allowed_types is not None and result_type not in allowed_types and not empty_bbox_fallback:
            self._add_validation_issue(issues, "unexpected type=%s expected=%s" %
                                       (result_type, '/'.join(allowed_types)))

        if result_type == 'bbox':
            self._validate_bbox_result(mode_name, parsed, issues)
        elif result_type == 'quad':
            self._validate_quad_result(mode_name, parsed, issues)
        elif result_type == 'center':
            self._validate_center_result(mode_name, parsed, issues)
        elif result_type == 'string':
            self._validate_string_result(parsed, issues)
        elif result_type == 'string_list':
            self._validate_string_list_result(parsed, issues)
        elif result_type == 'ocr':
            self._validate_ocr_result(mode_name, parsed, issues)
        elif result_type == 'color':
            self._validate_color_result(parsed, issues)
        elif result_type == 'multi_color':
            self._validate_multi_color_result(parsed, issues)
        elif result_type == 'line':
            self._validate_line_result(parsed, issues)
        elif result_type == 'keypoint':
            self._validate_keypoint_result(parsed, issues)
        elif result_type == 'hand_kp':
            self._validate_hand_kp_result(parsed, issues)
        elif result_type == 'face_kp':
            self._validate_face_kp_result(parsed, issues)

        if issues:
            self.last_detect_validation = {
                'mode': mode_name,
                'type': result_type,
                'result': parsed,
                'issues': list(issues),
            }
        else:
            self.last_detect_validation = None
        self._log_validation_issues(mode_name, result_type, issues)

    def _emit_detect_result(self, parsed, raw_values=None):
        self._validate_detect_result(parsed)
        self.last_detect_raw_values = list(raw_values) if isinstance(raw_values, list) else raw_values
        self.last_detect_raw_text = self._format_detect_raw_values(parsed, self.last_detect_raw_values)
        if self.on_detect_result:
            self.on_detect_result(parsed)

    # ========================================================================
    # 便捷命令方法 - 系统控制
    # ========================================================================

    def set_mode(self, mode):
        """设置运行模式"""
        mode_idx = self._normalize_mode_index(mode)
        current_mode_idx = self._current_mode_index_for_switch()
        empty_mode_idx = APP_NAME_TO_INDEX.get('Empty', 0)

        if mode_idx is None:
            return False

        if (mode_idx != empty_mode_idx and
                current_mode_idx not in (None, empty_mode_idx, mode_idx)):
            if not self._set_mode_and_confirm_mode(empty_mode_idx, timeout_ms=5000):
                self._log("旧应用切空未确认，取消进入新应用: from=%s, to=%s" % (
                    APP_INDEX_TO_NAME.get(current_mode_idx, 'Unknown(%s)' % current_mode_idx),
                    APP_INDEX_TO_NAME.get(mode_idx, 'Unknown(%s)' % mode_idx),
                ))
                return False

        return self._set_mode_and_confirm_mode(mode_idx, timeout_ms=5000)

    def set_volume(self, volume):
        """设置音量 (0-100)"""
        payload = CompactCodec.encode_uint8(volume)
        success, _ = self.send_command_and_wait(CMD_SET_VOLUME, payload)
        return success

    def set_wifi(self, ssid, password):
        """配置WiFi - 同步等待连接结果"""
        payload = CompactCodec.encode_string(ssid) + CompactCodec.encode_string(password)
        success, result = self.send_command_and_wait_async_result(CMD_SET_WIFI, payload, wait_cmd=CMD_EMPTY_RETURN,
                                                                  timeout_ms=15000)
        if not success:
            return False, None

        raw_flag = None
        if isinstance(result, (bytes, bytearray)) and len(result) > 0:
            raw_flag = result[0]
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            raw_flag = result[0]

        if raw_flag in (0, 1):
            connected = bool(raw_flag)
            return connected, {'connected': connected, 'raw': raw_flag}

        return True, result

    def request_status(self, timeout_ms=3000):
        """请求状态。

        默认实现按“同步命令结果”等待；像 wonderlens_system UART 这类
        直接返回心跳/结果上报的子类应覆盖本方法。
        """
        return self.send_command_and_wait(CMD_REQUEST_STATUS, timeout_ms=timeout_ms)

    def clear_memory(self):
        """清空 wonderlens_system 结果/异步缓存。"""
        success, _ = self.send_command_and_wait(CMD_CLEAR_MEMORY)
        return success

    def get_protocol_info(self):
        """获取协议版本/能力/当前最大帧长度。"""
        success, result = self.send_command_and_wait(CMD_GET_PROTOCOL_INFO)
        if not success or not result or len(result) < 10:
            return None
        return self._decode_protocol_info(result[2:])

    # ========================================================================
    # 便捷命令方法 - 检测参数
    # ========================================================================

    def set_confidence_threshold(self, threshold):
        """设置置信度阈值 (0-100)"""
        payload = CompactCodec.encode_uint8(threshold)
        success, _ = self.send_command_and_wait(CMD_SET_CONF_THRESH, payload)
        return success

    def set_nms_threshold(self, threshold):
        """设置NMS阈值 (0-100)"""
        payload = CompactCodec.encode_uint8(threshold)
        success, _ = self.send_command_and_wait(CMD_SET_NMS_THRESH, payload)
        return success

    def set_mask_threshold(self, threshold):
        """设置分割掩码阈值 (0-100)"""
        payload = CompactCodec.encode_uint8(threshold)
        success, _ = self.send_command_and_wait(CMD_SEG_SET_MASK_THRESH, payload)
        return success

    def set_run_enabled(self, enabled):
        """设置是否运行检测"""
        payload = CompactCodec.encode_uint8(0 if enabled else 1)
        success, _ = self.send_command_and_wait(CMD_DISABLE_RUN, payload)
        return bool(success)

    def set_simple_result_mode(self, enabled):
        """设置精简结果返回模式"""
        payload = CompactCodec.encode_uint8(1 if enabled else 0)
        success, _ = self.send_command_and_wait(CMD_SET_SIMPLE_RESULT, payload)
        return success

    # ========================================================================
    # 便捷命令方法 - 人脸相关
    # ========================================================================

    def _normalize_face_db_mode_name(self, mode):
        if isinstance(mode, str):
            mode_name = mode
        else:
            mode_name = APP_INDEX_TO_NAME.get(mode)

        if mode_name not in FACE_DB_MODE_NAMES:
            self._log("人脸特征库命令仅支持模式: %s, 当前=%s" %
                      ("/".join(FACE_DB_MODE_NAMES), mode))
            return None
        return mode_name

    def _switch_to_face_db_mode(self, mode):
        mode_name = self._normalize_face_db_mode_name(mode)
        if mode_name is None:
            return False
        return self.set_mode(mode_name)

    def face_learn_in_mode(self, mode, face_id):
        """切换到指定人脸模式后执行学习。"""
        if not self._switch_to_face_db_mode(mode):
            return False
        return self.face_learn(face_id)

    def face_enhance_learn_in_mode(self, mode, face_id):
        """切换到指定人脸模式后执行加强学习。"""
        if not self._switch_to_face_db_mode(mode):
            return False
        return self.face_enhance_learn(face_id)

    def face_learn_at_point_in_mode(self, mode, x, y, name):
        """切换到指定人脸模式后按坐标学习。"""
        if not self._switch_to_face_db_mode(mode):
            return False
        return self.face_learn_at_point(x, y, name)

    def face_delete_in_mode(self, mode, face_id):
        """切换到指定人脸模式后删除样本。"""
        if not self._switch_to_face_db_mode(mode):
            return False
        return self.face_delete(face_id)

    def face_rename_in_mode(self, mode, old_name, new_name):
        """切换到指定人脸模式后重命名样本。"""
        if not self._switch_to_face_db_mode(mode):
            return False
        return self.face_rename(old_name, new_name)

    def face_pose_learn(self, face_id):
        return self.face_learn_in_mode('FacePose', face_id)

    def face_pose_enhance_learn(self, face_id):
        return self.face_enhance_learn_in_mode('FacePose', face_id)

    def face_pose_learn_at_point(self, x, y, name):
        return self.face_learn_at_point_in_mode('FacePose', x, y, name)

    def face_pose_delete(self, face_id):
        return self.face_delete_in_mode('FacePose', face_id)

    def face_pose_rename(self, old_name, new_name):
        return self.face_rename_in_mode('FacePose', old_name, new_name)

    def eye_gaze_learn(self, face_id):
        return self.face_learn_in_mode('EyeGaze', face_id)

    def eye_gaze_enhance_learn(self, face_id):
        return self.face_enhance_learn_in_mode('EyeGaze', face_id)

    def eye_gaze_learn_at_point(self, x, y, name):
        return self.face_learn_at_point_in_mode('EyeGaze', x, y, name)

    def eye_gaze_delete(self, face_id):
        return self.face_delete_in_mode('EyeGaze', face_id)

    def eye_gaze_rename(self, old_name, new_name):
        return self.face_rename_in_mode('EyeGaze', old_name, new_name)

    def face_learn(self, face_id):
        """学习人脸"""
        payload = CompactCodec.encode_string(face_id)
        success, _ = self.send_command_and_wait(CMD_FACE_LEARN, payload, timeout_ms=5000)
        return success

    def face_enhance_learn(self, face_id):
        """加强学习人脸

        如果名称尚未存在，则创建新的人脸条目；
        如果名称已存在，则为该名称追加样本以加强识别效果。
        """
        payload = CompactCodec.encode_string(face_id)
        success, _ = self.send_command_and_wait(CMD_FACE_ENHANCE_LEARN, payload, timeout_ms=5000)
        return success

    def face_learn_at_point(self, x, y, name):
        """按给定坐标选择最近人脸并学习到指定名称"""
        payload = (
            CompactCodec.encode_uint32(x) +
            CompactCodec.encode_uint32(y) +
            CompactCodec.encode_string(name)
        )
        success, _ = self.send_command_and_wait(CMD_FACE_LEARN_AT_POINT, payload, timeout_ms=5000)
        return success

    def face_delete(self, face_id):
        """删除人脸"""
        payload = CompactCodec.encode_string(face_id)
        success, _ = self.send_command_and_wait(CMD_FACE_DELETE, payload)
        return success

    def face_rename(self, old_name, new_name):
        """重命名人脸"""
        payload = CompactCodec.encode_string(old_name) + CompactCodec.encode_string(new_name)
        success, _ = self.send_command_and_wait(CMD_FACE_RENAME, payload)
        return success

    def set_face_recognition_threshold(self, threshold):
        """设置人脸识别阈值 (0-100)"""
        payload = CompactCodec.encode_uint8(threshold)
        success, _ = self.send_command_and_wait(CMD_FACE_SET_RECOG_CONF, payload)
        return success

    def set_face_high_precision(self, enabled):
        """设置人脸高精度模式"""
        payload = CompactCodec.encode_uint8(1 if enabled else 0)
        success, _ = self.send_command_and_wait(CMD_FACE_HIGH_PRECISION, payload)
        return success

    def set_face_keypoint_mode(self, enabled):
        """设置人脸关键点模式
        
        启用后只返回五官关键点数据，不进行人脸比对识别
        
        Args:
            enabled: True启用关键点模式，False禁用
        """
        payload = CompactCodec.encode_uint8(1 if enabled else 0)
        success, _ = self.send_command_and_wait(CMD_FACE_ENABLE_KEYPOINT, payload)
        return success

    def set_face_detect_only_mode(self, enabled):
        """设置只检测人脸模式
        
        启用后只进行人脸检测，返回多个人脸的xywh坐标，加快检测速度
        
        Args:
            enabled: True启用只检测模式，False禁用
        """
        payload = CompactCodec.encode_uint8(1 if enabled else 0)
        success, _ = self.send_command_and_wait(CMD_FACE_DETECT_ONLY, payload)
        return success

    def person_kp_learn(self, name):
        """学习人体关键点特征"""
        payload = CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_PERSON_KP_LEARN, payload, timeout_ms=5000)
        return success

    def person_kp_enhance_learn(self, name):
        """加强学习人体关键点特征

        如果名称尚未存在，则创建新的人体关键点条目；
        如果名称已存在，则为该名称追加样本以加强识别效果。
        """
        payload = CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_PERSON_KP_ENHANCE_LEARN, payload, timeout_ms=5000)
        return success

    def person_kp_delete(self, name):
        """删除人体关键点特征"""
        payload = CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_PERSON_KP_DELETE, payload)
        return success

    def person_kp_rename(self, old_name, new_name):
        """重命名人体关键点特征"""
        payload = CompactCodec.encode_string(old_name) + CompactCodec.encode_string(new_name)
        success, _ = self.send_command_and_wait(CMD_PERSON_KP_RENAME, payload)
        return success

    def hand_kp_learn(self, name):
        """学习手掌关键点特征"""
        payload = CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_HAND_KP_LEARN, payload, timeout_ms=5000)
        return success

    def hand_kp_enhance_learn(self, name):
        """加强学习手掌关键点特征

        如果名称尚未存在，则创建新的手掌关键点条目；
        如果名称已存在，则为该名称追加样本以加强识别效果。
        """
        payload = CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_HAND_KP_ENHANCE_LEARN, payload, timeout_ms=5000)
        return success

    def hand_kp_delete(self, name):
        """删除手掌关键点特征"""
        payload = CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_HAND_KP_DELETE, payload)
        return success

    def hand_kp_rename(self, old_name, new_name):
        """重命名手掌关键点特征"""
        payload = CompactCodec.encode_string(old_name) + CompactCodec.encode_string(new_name)
        success, _ = self.send_command_and_wait(CMD_HAND_KP_RENAME, payload)
        return success

    # ========================================================================
    # 便捷命令方法 - 颜色检测
    # ========================================================================

    def set_color_target(self, color_name):
        """设置目标颜色"""
        payload = CompactCodec.encode_string(color_name)
        success, _ = self.send_command_and_wait(CMD_COLOR_SET_TARGET, payload)
        return success

    def select_color_learning_profile(self, name):
        """在颜色学习模式下按名称选中颜色项"""
        return self.set_color_target(name)

    def set_color_threshold(self, threshold_dict):
        """设置颜色LAB阈值"""
        payload = bytearray([len(threshold_dict)])
        for color_name, thresh in threshold_dict.items():
            payload.extend(CompactCodec.encode_string(color_name))
            for v in thresh:
                payload.append(v & 0xFF)
        success, _ = self.send_command_and_wait(CMD_COLOR_SET_THRESH, bytes(payload))
        return success

    def get_color_threshold(self, color_name):
        """获取颜色LAB阈值 - 同步等待返回"""
        payload = CompactCodec.encode_string(color_name)
        success, result = self.send_command_and_wait(CMD_COLOR_GET_THRESH, payload)
        if success and result and len(result) > 2:
            # 结果格式: [cmd][code][lab_thresh 6字节]
            extra = result[2:]
            if len(extra) >= 6:
                values = list(extra[:6])
                for index in range(2, 6):
                    if values[index] >= 128:
                        values[index] -= 256
                return values
        return None

    def set_color_filter(self, number, min_area, max_area):
        """设置颜色滤波参数"""
        payload = bytes([number]) + struct.pack('>I', min_area) + struct.pack('>I', max_area)
        success, _ = self.send_command_and_wait(CMD_COLOR_SET_FILTER, payload)
        return success

    def set_color_min_area(self, min_area):
        """设置颜色最小检测面积"""
        payload = struct.pack('>I', min_area)
        success, _ = self.send_command_and_wait(CMD_COLOR_SET_MIN_AREA, payload)
        return success

    def set_multi_color_list(self, colors):
        """设置多颜色检测列表"""
        payload = bytearray([len(colors)])
        for color in colors:
            payload.extend(CompactCodec.encode_string(color))
        success, _ = self.send_command_and_wait(CMD_MULTI_COLOR_SET_LIST, bytes(payload))
        return success

    def set_line_roi(self, roi_list):
        """设置线检测ROI"""
        if len(roi_list) != 3:
            return False
        payload = CompactCodec.encode_line_roi(roi_list)
        success, _ = self.send_command_and_wait(CMD_LINE_SET_ROI, payload)
        return success

    def set_color_learning_point(self, x, y, name):
        """设置颜色学习取样点"""
        payload = struct.pack('>HH', x, y) + CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_COLOR_LEARNING_SET_POINT, payload)
        return success

    def save_color_learning(self, name):
        """保存颜色学习颜色项（0x48）。"""
        payload = CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_COLOR_LEARNING_SAVE, payload)
        return success

    def rename_color_learning(self, old_name, new_name):
        """重命名颜色学习颜色项"""
        payload = CompactCodec.encode_string(old_name) + CompactCodec.encode_string(new_name)
        success, _ = self.send_command_and_wait(CMD_COLOR_LEARNING_RENAME, payload)
        return success

    def delete_color_learning(self, name):
        """删除颜色学习颜色项"""
        payload = CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_COLOR_LEARNING_DELETE, payload)
        return success

    # ========================================================================
    # 便捷命令方法 - 自学习/跟踪
    # ========================================================================

    def selflearn_set_name(self, name):
        """设置自学习目标名称并开始"""
        payload = CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_SELFLEARN_SET_NAME, payload)
        return success

    def selflearn_set_rect(self, x, y, w, h):
        """设置自学习样本区域"""
        payload = CompactCodec.encode_bbox(x, y, w, h)
        success, _ = self.send_command_and_wait(CMD_SELFLEARN_SET_RECT, payload)
        return success

    def selflearn_set_frame(self, frame_count):
        """设置每个特征学习的帧数"""
        payload = CompactCodec.encode_uint16(frame_count)
        success, _ = self.send_command_and_wait(CMD_SELFLEARN_SET_FRAME, payload)
        return success

    def selflearn_set_features(self, feature_count):
        """设置自学习特征数"""
        payload = CompactCodec.encode_uint8(feature_count)
        success, _ = self.send_command_and_wait(CMD_SELFLEARN_SET_FEATURES, payload)
        return success

    def selflearn_delete(self, name):
        """删除指定名称的特征"""
        payload = CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_SELFLEARN_DELETE, payload)
        return success

    def selflearn_rename(self, old_name, new_name):
        """重命名指定名称的自学习特征"""
        payload = CompactCodec.encode_string(old_name) + CompactCodec.encode_string(new_name)
        success, _ = self.send_command_and_wait(CMD_SELFLEARN_RENAME, payload)
        return success

    def nanotrack_set_rect(self, x, y, w, h):
        """设置NanoTrack初始框并开始跟踪"""
        payload = CompactCodec.encode_bbox(x, y, w, h)
        success, _ = self.send_command_and_wait(CMD_NANOTRACK_SET_RECT, payload)
        return success

    def nanotrack_stop(self, stop=True):
        """停止NanoTrack跟踪"""
        payload = CompactCodec.encode_uint8(1 if stop else 0)
        success, _ = self.send_command_and_wait(CMD_NANOTRACK_STOP, payload)
        return success

    def set_gesture_frame_count(self, frame_count):
        """设置动态手势检测帧数"""
        payload = CompactCodec.encode_uint16(frame_count)
        success, _ = self.send_command_and_wait(CMD_GESTURE_SET_FRAME, payload)
        return success

    def dgesture_ctrl(self, action, name=None, name2=None, timeout_ms=3000):
        """发送动态手势控制指令"""
        payload = CompactCodec.encode_uint8(action)
        if name is not None:
            payload += CompactCodec.encode_string(name)
        if name2 is not None:
            payload += CompactCodec.encode_string(name2)
        success, _ = self.send_command_and_wait(CMD_DGESTURE_CTRL, payload, timeout_ms=timeout_ms)
        return success

    def dgesture_record_start(self):
        """开始录制动态手势"""
        return self.dgesture_ctrl(DGESTURE_ACTION_RECORD_START)

    def dgesture_record_stop(self):
        """停止录制动态手势"""
        return self.dgesture_ctrl(DGESTURE_ACTION_RECORD_STOP)

    def dgesture_save(self, name):
        """保存动态手势

        如果名称不存在，则创建新条目；
        如果名称已存在，则覆盖该名称的已有样本。
        """
        return self.dgesture_ctrl(DGESTURE_ACTION_SAVE, name=name, timeout_ms=5000)

    def dgesture_enhance_save(self, name):
        """加强保存动态手势

        如果名称尚未保存，则创建新的动态手势条目；
        如果名称已存在，则为该名称追加样本以加强识别效果。
        """
        payload = CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_DGESTURE_ENHANCE_SAVE, payload, timeout_ms=5000)
        return success

    def dgesture_enhance_save_drop_oldest(self, name):
        """加强保存动态手势，满额时自动丢弃最旧样本"""
        return self.dgesture_ctrl(DGESTURE_ACTION_SAVE_APPEND_DROP_OLDEST,
                                  name=name,
                                  timeout_ms=5000)

    def dgesture_delete(self, name):
        """删除动态手势"""
        return self.dgesture_ctrl(DGESTURE_ACTION_DELETE, name=name)

    def dgesture_rename(self, old_name, new_name):
        """重命名动态手势"""
        return self.dgesture_ctrl(DGESTURE_ACTION_RENAME, name=old_name, name2=new_name)

    def set_hand_detect_only_mode(self, enabled):
        """设置只检测手掌模式

        启用后只进行手掌检测，返回多个手掌的xywh坐标，加快检测速度

        Args:
            enabled: True启用只检测模式，False禁用
        """
        payload = CompactCodec.encode_uint8(1 if enabled else 0)
        success, _ = self.send_command_and_wait(CMD_HAND_DETECT_ONLY, payload)
        return success

    def set_custom_model(self, model_index):
        """设置自定义检测模型索引"""
        payload = CompactCodec.encode_uint32(model_index)
        success, _ = self.send_command_and_wait(CMD_CUSTOM_SET_MODEL, payload)
        return success

    def set_object_mode(self, mode):
        """设置物体识别模式。

        Args:
            mode: 0/'detect' 表示检测，1/'cls' 表示分类，2/'seg' 表示分割
        """
        mode_map = {
            'detect': 0,
            'detection': 0,
            'cls': 1,
            'classification': 1,
            'classify': 1,
            'seg': 2,
            'segment': 2,
            'segmentation': 2,
        }
        if isinstance(mode, str):
            mode_key = mode.strip().lower()
            mode_value = mode_map.get(mode_key)
            if mode_value is None:
                self._log("无效物体模式: %s (支持 detect/cls/seg 或 0/1/2)" % mode)
                return False
        else:
            try:
                mode_value = int(mode)
            except (TypeError, ValueError):
                self._log("无效物体模式: %s (支持 detect/cls/seg 或 0/1/2)" % mode)
                return False
            if mode_value not in (0, 1, 2):
                self._log("无效物体模式值: %s (仅支持 0/1/2)" % mode_value)
                return False

        payload = CompactCodec.encode_uint8(mode_value)
        success, _ = self.send_command_and_wait(CMD_OBJECT_SET_MODE, payload)
        return success

    def set_face_pose_threshold(self, roll, pitch, yaw):
        """设置人脸姿态/注视匹配阈值。

        Args:
            roll: 横滚角阈值(1-180)
            pitch: 俯仰角阈值(1-180)
            yaw: 偏航角阈值(1-180)
        """
        values = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
        encoded_values = {}
        for key, value in values.items():
            try:
                ivalue = int(value)
            except (TypeError, ValueError):
                self._log("无效姿态阈值 %s=%s (要求 1-180)" % (key, value))
                return False
            if ivalue < 1 or ivalue > 180:
                self._log("姿态阈值越界 %s=%s (要求 1-180)" % (key, ivalue))
                return False
            encoded_values[key] = ivalue

        payload = (
            CompactCodec.encode_uint8(encoded_values['roll']) +
            CompactCodec.encode_uint8(encoded_values['pitch']) +
            CompactCodec.encode_uint8(encoded_values['yaw'])
        )
        success, _ = self.send_command_and_wait(CMD_FACE_SET_POSE_THRESH, payload)
        return success

    def set_eye_gaze_threshold(self, roll, pitch, yaw):
        """设置注视估计模板匹配阈值（复用 0x29 指令）。

        该接口要求设备当前模式为 EyeGaze。
        """
        return self.set_face_pose_threshold(roll, pitch, yaw)

    def media_camera_snapshot(self):
        """触发媒体相机拍照并按当前命名规则保存。"""
        success, _ = self.send_command_and_wait(CMD_MEDIA_CAMERA_SNAPSHOT, b'')
        return success

    def media_set_photo_prefix(self, prefix):
        """设置照片名前缀；传空字符串可清除前缀并恢复默认命名。"""
        payload = CompactCodec.encode_string(prefix or '')
        success, _ = self.send_command_and_wait(CMD_MEDIA_SET_PHOTO_PREFIX, payload)
        return success

    def media_delete_photo(self, name):
        """按名称删除指定照片；支持传基础名或完整文件名。"""
        payload = CompactCodec.encode_string(name)
        success, _ = self.send_command_and_wait(CMD_MEDIA_DELETE_PHOTO, payload)
        return success

    def media_enter_camera_app(self):
        """切换设备 UI 到媒体相机 app。"""
        success, _ = self.send_command_and_wait(CMD_MEDIA_ENTER_CAMERA_APP, b'')
        return success

    def media_set_photo_start(self, start_value):
        """设置拍照命名前缀的起始值。0 表示从最小缺失号补位。"""
        payload = CompactCodec.encode_uint32(start_value)
        success, _ = self.send_command_and_wait(CMD_MEDIA_SET_PHOTO_START, payload)
        return success

    # ========================================================================
    # 便捷命令方法 - 语音交互
    # ========================================================================

    def set_llm_key(self, api_key):
        """设置LLM API密钥"""
        payload = CompactCodec.encode_string(api_key)
        success, _ = self.send_command_and_wait(CMD_SET_LLM_KEY, payload)
        return success

    def set_llm_model(self, model):
        """设置LLM模型"""
        payload = CompactCodec.encode_string(model)
        success, _ = self.send_command_and_wait(CMD_SET_LLM_MODEL, payload)
        return success

    def set_vlm_model(self, model):
        """设置VLM模型"""
        payload = CompactCodec.encode_string(model)
        success, _ = self.send_command_and_wait(CMD_SET_VLM_MODEL, payload)
        return success

    def set_llm_base_url(self, base_url):
        """设置LLM服务地址"""
        payload = CompactCodec.encode_string(base_url)
        success, _ = self.send_command_and_wait(CMD_SET_LLM_BASE_URL, payload)
        return success

    def set_vlm_base_url(self, base_url):
        """设置VLM服务地址"""
        payload = CompactCodec.encode_string(base_url)
        success, _ = self.send_command_and_wait(CMD_SET_VLM_BASE_URL, payload)
        return success

    def set_speech_url(self, base_url):
        """设置语音服务地址"""
        payload = CompactCodec.encode_string(base_url)
        success, _ = self.send_command_and_wait(CMD_SET_SPEECH_URL, payload)
        return success

    def set_tts_voice(self, model, voice):
        """设置TTS音色"""
        payload = CompactCodec.encode_string(model) + CompactCodec.encode_string(voice)
        success, _ = self.send_command_and_wait(CMD_SET_TTS_VOICE, payload)
        return success

    def set_asr_language(self, language):
        """设置ASR语言"""
        payload = CompactCodec.encode_string(language)
        success, _ = self.send_command_and_wait(CMD_SET_ASR_LANG, payload)
        return success

    def set_system_prompt(self, prompt):
        """设置系统提示词"""
        payload = CompactCodec.encode_string(prompt)
        success, _ = self.send_command_and_wait(CMD_SET_PROMPT, payload)
        return success

    def set_thinking_mode(self, enabled):
        """启用/禁用思考模式"""
        payload = CompactCodec.encode_uint8(1 if enabled else 0)
        success, _ = self.send_command_and_wait(CMD_SET_THINKING, payload)
        return success

    def set_search_mode(self, enabled):
        """启用/禁用搜索模式"""
        payload = CompactCodec.encode_uint8(1 if enabled else 0)
        success, _ = self.send_command_and_wait(CMD_SET_SEARCH, payload)
        return success

    def set_start_silence(self, silence_ms):
        """设置开始语句静音时间(毫秒)"""
        payload = CompactCodec.encode_uint16(silence_ms)
        success, _ = self.send_command_and_wait(CMD_SET_START_SILENCE, payload)
        return success

    def set_end_silence(self, silence_ms):
        """设置结束语句静音时间(毫秒)"""
        payload = CompactCodec.encode_uint16(silence_ms)
        success, _ = self.send_command_and_wait(CMD_SET_END_SILENCE, payload)
        return success

    def start_asr(self, start=True):
        """开启/关闭语音识别 - 同步等待结果"""
        payload = CompactCodec.encode_uint8(1 if start else 0)
        success, result = self.send_command_and_wait_async_result(CMD_ASR, payload, timeout_ms=CMD_ASR_TIMEOUT_MS)
        return success, result

    def tts_speak(self, text):
        """文本转语音 - 同步等待播放完成"""
        payload = CompactCodec.encode_string(text)
        success, result = self.send_command_and_wait_async_result(CMD_TTS, payload, wait_cmd=CMD_EMPTY_RETURN)
        return success, result

    def llm_chat(self, message):
        """LLM对话 - 同步等待结果"""
        payload = CompactCodec.encode_string(message)
        success, result = self.send_command_and_wait_async_result(CMD_LLM_CHAT, payload)
        return success, result

    def vlm_chat(self, message):
        """VLM对话(带视觉) - 同步等待结果"""
        payload = CompactCodec.encode_string(message)
        success, result = self.send_command_and_wait_async_result(CMD_VLM_CHAT, payload)
        return success, result

    def set_mcp_tools(self, tools):
        """设置MCP工具"""
        payload = data_pack(tools)
        success, _ = self.send_command_and_wait(CMD_SET_MCP_TOOLS, payload)
        return success

    def send_mcp_result(self, result):
        """发送MCP执行结果"""
        payload = data_pack(result)
        success, _ = self.send_command_and_wait(CMD_RESULT_RETURN, payload)
        return success

    # ========================================================================
    # 帧处理方法
    # ========================================================================

    def _handle_frame(self, frame):
        """处理单个帧"""
        self._note_rx_frame_activity(frame.frame_type,
                                     frame.func_code,
                                     len(frame.payload) if frame.payload else 0)
        try:
            if frame.continuation or self.reassembler.in_progress:
                payload = self.reassembler.feed(frame)
                if payload is None:
                    return
            else:
                payload = frame.payload
        except Exception as e:
            self.reassembler.reset()
            self._log("Fragment reassemble error: %s" % e)
            return

        if frame.frame_type == FRAME_TYPE_RPT:
            self._handle_report(frame.func_code, payload)
        elif frame.frame_type == FRAME_TYPE_RSP:
            self._handle_response(frame.func_code, frame.txn_id, payload)

    def _handle_report(self, func_code, payload):
        """处理上报帧"""
        self._note_report_activity(func_code, len(payload) if payload else 0)
        if func_code == RPT_HEARTBEAT:
            self._handle_heartbeat(payload)
        elif func_code == RPT_ERROR:
            self._handle_error(payload)
        elif func_code == RPT_DETECT_BBOX:
            self._handle_detect_bbox(payload)
        elif func_code == RPT_DETECT_STR:
            self._handle_detect_str(payload)
        elif func_code == RPT_DETECT_OCR:
            self._handle_detect_ocr(payload)
        elif func_code == RPT_DETECT_COLOR:
            self._handle_detect_color(payload)
        elif func_code == RPT_DETECT_LINE:
            self._handle_detect_line(payload)
        elif func_code == RPT_DETECT_KEYPOINT:
            self._handle_detect_keypoint(payload)
        elif func_code == RPT_DETECT_HAND_KP:
            self._handle_detect_hand_kp(payload)
        elif func_code == RPT_DETECT_CENTER:
            self._handle_detect_center(payload)
        elif func_code == RPT_DETECT_FACE_KP:
            self._handle_detect_face_kp(payload)
        elif func_code == RPT_DETECT_QUAD:
            self._handle_detect_quad(payload)
        else:
            self._log("Unknown report: func=0x%02X" % func_code)

    def _handle_heartbeat(self, payload):
        """处理心跳包"""
        if not payload:
            return
        try:
            self.current_heartbeat = HeartbeatData.decode(payload)
            self._current_mode = self.current_heartbeat.mode
            self._current_mode_name = APP_INDEX_TO_NAME.get(self._current_mode, 'Unknown(%s)' % self._current_mode)
            self._heartbeat_serial += 1

            if self.on_heartbeat:
                self.on_heartbeat(self.current_heartbeat.to_dict())
        except Exception as e:
            self._log("Heartbeat parse error: %s" % e)

    def _decode_protocol_info(self, extra):
        if not extra or len(extra) < 8:
            return extra
        try:
            capability_flags = struct.unpack('>I', extra[2:6])[0]
            max_frame_len = struct.unpack('>H', extra[6:8])[0]
        except Exception:
            return extra
        return {
            'major': extra[0],
            'minor': extra[1],
            'capability_flags': capability_flags,
            'max_frame_len': max_frame_len,
        }

    def _handle_response(self, func_code, txn_id, payload):
        """处理响应帧"""
        self._note_response_activity(func_code, len(payload) if payload else 0)
        if not payload or len(payload) < RSP_ERROR_PREFIX_LEN:
            return
        try:
            cmd = func_code
            code = payload[0]
            error_module = payload[1]
            error_subcode = ((payload[2] << 8) | payload[3]) & 0xFFFF
            extra = payload[RSP_ERROR_PREFIX_LEN:] if len(payload) > RSP_ERROR_PREFIX_LEN else None
            result_data = bytes([
                cmd,
                code,
                error_module,
                (error_subcode >> 8) & 0xFF,
                error_subcode & 0xFF,
            ]) + (extra or b'')
            callback_extra = extra

            if cmd == CMD_GET_PROTOCOL_INFO:
                callback_extra = self._decode_protocol_info(extra)
            elif cmd == CMD_RESULT_RETURN and extra:
                try:
                    result_obj, _ = data_unpack(extra)
                    if code == ERR_OK and self.on_mcp_result:
                        self.on_mcp_result(result_obj)
                    if code == ERR_OK and isinstance(result_obj, dict):
                        if 'asr' in result_obj and self.on_asr_result:
                            self.on_asr_result(result_obj)
                        elif 'llm' in result_obj and self.on_llm_result:
                            self.on_llm_result(result_obj)
                        elif 'vlm' in result_obj and self.on_vlm_result:
                            self.on_vlm_result(result_obj)
                        elif self.on_llm_result:
                            self.on_llm_result(result_obj)
                    elif code == ERR_OK and isinstance(result_obj, str):
                        if self.on_llm_result:
                            self.on_llm_result({'text': result_obj})
                    callback_extra = result_obj
                except Exception as e:
                    self._log("data_unpack error for cmd 0x%02X: %s" % (cmd, e))

            elif cmd == CMD_EMPTY_RETURN:
                result_flag = extra[0] if extra else None
                expected_owner = (
                    self._waiting_async_owner_cmd
                    if self._waiting_async_cmd == CMD_EMPTY_RETURN and txn_id == self._waiting_async_txn
                    else None
                )

                if expected_owner == CMD_TTS and self.on_tts_finish:
                    self.on_tts_finish({
                        'success': code == ERR_OK,
                        'raw': extra,
                    })
                elif expected_owner == CMD_SET_WIFI and self.on_wifi_connected:
                    self.on_wifi_connected({
                        'success': code == ERR_OK,
                        'connected': bool(result_flag),
                        'raw': result_flag,
                    })
                elif result_flag in (0, 1) and self.on_wifi_connected:
                    self.on_wifi_connected({
                        'success': code == ERR_OK,
                        'connected': bool(result_flag),
                        'raw': result_flag,
                    })
                elif (result_flag == 2 or result_flag is None) and self.on_tts_finish:
                    self.on_tts_finish({
                        'success': code == ERR_OK,
                        'raw': extra,
                    })

            success = (code == ERR_OK)

            if (
                self._waiting_async_cmd is not None and
                self._waiting_async_txn == txn_id and
                cmd == self._waiting_async_cmd
            ):
                self._async_result = callback_extra
                self._async_result_ready = True
                self._async_success = success

            if (
                self._pending_cmd is not None and
                self._pending_txn == txn_id and
                cmd == self._pending_cmd
            ):
                self._cmd_success = success
                self._cmd_result_data = result_data
                self._last_response_txn = txn_id
                self._pending_cmd = None
                self._pending_txn = None

            if self.on_command_result:
                self.on_command_result({
                    'cmd': cmd,
                    'txn': txn_id,
                    'code': code,
                    'code_name': ERROR_NAMES.get(code, 'Unknown(%s)' % code),
                    'module': error_module,
                    'module_name': ERROR_MODULE_NAMES.get(error_module, 'unknown'),
                    'subcode': error_subcode,
                    'success': success,
                    'extra': callback_extra
                })
        except Exception as e:
            self._log("Command result parse error: %s" % e)

    def _handle_error(self, payload):
        """处理错误上报"""
        if not payload or len(payload) < 4:
            return
        try:
            error_code = payload[0]
            error_module = payload[1]
            error_subcode = ((payload[2] << 8) | payload[3]) & 0xFFFF
            module_class_name = ERROR_MODULE_NAMES.get(error_module, 'unknown')
            offset = 4
            module_name = ""
            error_msg = ""
            if len(payload) > offset:
                try:
                    module_name, consumed = CompactCodec.decode_string(payload, offset)
                    offset += consumed
                    if offset < len(payload):
                        error_msg, _ = CompactCodec.decode_string(payload, offset)
                except Exception:
                    error_msg = payload[offset:].decode('utf-8', 'ignore')

            if self.on_error:
                self.on_error({
                    'code': error_code,
                    'code_name': ERROR_NAMES.get(error_code, 'Unknown(%s)' % error_code),
                    'module_id': error_module,
                    'module_class': module_class_name,
                    'subcode': error_subcode,
                    'module': module_name,
                    'message': error_msg
                })
            else:
                self._log("错误上报: module=%s class=%s sub=0x%04X code=0x%02X(%s), message=%s" % (
                    module_name or '--',
                    module_class_name,
                    error_subcode,
                    error_code,
                    ERROR_NAMES.get(error_code, 'Unknown(%s)' % error_code),
                    error_msg or '--',
                ))
        except Exception as e:
            self._log("Error parse failed: %s" % e)

    # ========================================================================
    # 检测结果处理方法
    # ========================================================================

    def _handle_detect_bbox(self, payload):
        """处理bbox检测结果"""
        if not payload:
            return
        try:
            offset = 0
            count = payload[offset]
            offset += 1
            raw_values = [count]

            results = []
            for _ in range(count):
                if offset + 8 > len(payload):
                    break
                x, y, w, h = struct.unpack_from('>HHHH', payload, offset)
                offset += 8
                raw_values.extend((x, y, w, h))

                extra, offset = self._parse_detect_extras(payload, offset, raw_values=raw_values)

                results.append({'x': x, 'y': y, 'w': w, 'h': h, 'extra': extra})

            self._emit_detect_result({'type': 'bbox', 'results': results}, raw_values=raw_values)
        except Exception as e:
            self._log("Detect bbox parse error: %s" % e)

    def _parse_detect_extras(self, payload, offset, raw_values=None):
        extra = []
        if offset >= len(payload):
            return extra, offset

        extra_count = payload[offset]
        offset += 1
        if isinstance(raw_values, list):
            raw_values.append(extra_count)
        for _ in range(extra_count):
            if offset >= len(payload):
                break
            extra_type = payload[offset]
            offset += 1
            if isinstance(raw_values, list):
                raw_values.append(extra_type)
            if extra_type == 1:  # 字符串
                s, consumed, declared_len = self._decode_raw_string_values(payload, offset)
                extra.append(s)
                if isinstance(raw_values, list):
                    raw_values.append(declared_len)
                    raw_values.append(s)
                offset += consumed
            elif extra_type == 2:  # 定点浮点 x100
                if offset + 2 <= len(payload):
                    val = struct.unpack_from('>h', payload, offset)[0]
                    converted = val / 100.0
                    extra.append(converted)
                    if isinstance(raw_values, list):
                        raw_values.append(converted)
                    offset += 2
            elif extra_type == 3:  # 整数
                if offset + 2 <= len(payload):
                    val = struct.unpack_from('>h', payload, offset)[0]
                    extra.append(val)
                    if isinstance(raw_values, list):
                        raw_values.append(val)
                    offset += 2
            elif extra_type == 4:  # int16列表
                if offset < len(payload):
                    list_len = payload[offset]
                    offset += 1
                    if isinstance(raw_values, list):
                        raw_values.append(list_len)
                    values = []
                    for _ in range(list_len):
                        if offset + 2 > len(payload):
                            break
                        value = struct.unpack_from('>h', payload, offset)[0]
                        values.append(value)
                        if isinstance(raw_values, list):
                            raw_values.append(value)
                        offset += 2
                    extra.append(values)
        return extra, offset

    def _handle_detect_quad(self, payload):
        """处理旋转框检测结果"""
        if not payload:
            return
        try:
            offset = 0
            count = payload[offset]
            offset += 1
            raw_values = [count]

            results = []
            for _ in range(count):
                if offset + 16 > len(payload):
                    break
                points = []
                for _ in range(8):
                    value = struct.unpack_from('>h', payload, offset)[0]
                    points.append(value)
                    raw_values.append(value)
                    offset += 2
                extra, offset = self._parse_detect_extras(payload, offset, raw_values=raw_values)
                results.append({'points': points, 'extra': extra})

            self._emit_detect_result({'type': 'quad', 'results': results}, raw_values=raw_values)
        except Exception as e:
            self._log("Detect quad parse error: %s" % e)

    def _handle_detect_str(self, payload):
        """处理字符串结果"""
        if not payload:
            return
        try:
            # 兼容两种历史格式:
            # 1. 单字符串: [strlen][bytes...]
            # 2. 字符串列表: [count][strlen][bytes...]...
            # 当 payload[0] + 1 == len(payload) 时，优先按单字符串解析，
            # 否则像 "down" -> [4,'d','o','w','n'] 会被误当成 4 个空串。
            if len(payload) > 1 and payload[0] + 1 == len(payload) and not (payload[0] == 1 and payload[1] == 0):
                text, _, declared_len = self._decode_raw_string_values(payload, 0)
                texts = [text]
                raw_values = [declared_len, text]
            else:
                texts = []
                raw_values = []
                try:
                    count = payload[0]
                    offset = 1
                    raw_values.append(count)
                    for _ in range(count):
                        text, consumed, declared_len = self._decode_raw_string_values(payload, offset)
                        texts.append(text)
                        raw_values.append(declared_len)
                        raw_values.append(text)
                        offset += consumed
                    if offset != len(payload):
                        raise ValueError("count-prefixed string payload has trailing bytes")
                except Exception:
                    text, _, declared_len = self._decode_raw_string_values(payload, 0)
                    texts = [text]
                    raw_values = [declared_len, text]

            if len(texts) == 1:
                self._emit_detect_result({'type': 'string', 'value': texts[0]}, raw_values=raw_values)
            else:
                self._emit_detect_result({'type': 'string_list', 'results': texts}, raw_values=raw_values)
        except Exception as e:
            self._log("Detect string parse error: %s" % e)

    def _handle_detect_ocr(self, payload):
        """处理OCR/车牌结果"""
        if not payload or len(payload) < 2:
            return
        try:
            offset = 0
            count = payload[offset]
            offset += 1

            # 读取has_text标志位（服务端编码时添加的）
            has_text = payload[offset] if offset < len(payload) else 0
            offset += 1
            raw_values = [count, has_text]

            results = []
            for item_idx in range(count):
                if offset + 16 > len(payload):
                    break
                points = []
                truncated = False
                for _ in range(8):
                    pt = struct.unpack_from('>h', payload, offset)[0]
                    points.append(pt)
                    raw_values.append(pt)
                    offset += 2

                text = ""
                if has_text:
                    text, consumed, truncated, replaced, declared_len, actual_len = self._decode_prefixed_text_lossy(payload, offset)
                    if consumed <= 0:
                        self._log("Detect OCR text missing: idx=%d count=%d parsed=%d" %
                                  (item_idx, count, len(results)))
                        results.append({'points': points, 'text': text})
                        break
                    raw_values.append(declared_len)
                    raw_values.append(text)
                    offset += consumed
                    if replaced or truncated:
                        detail = []
                        if replaced:
                            detail.append("utf8_replaced")
                        if truncated:
                            detail.append("truncated declared=%d actual=%d" % (declared_len, actual_len))
                        self._log("Detect OCR text recovered: idx=%d %s" %
                                  (item_idx, ' '.join(detail)))

                results.append({'points': points, 'text': text})
                if has_text and truncated:
                    break

            parsed = {'type': 'ocr', 'results': results}
        except Exception as e:
            payload_hex = payload.hex()
            if len(payload_hex) > 96:
                payload_hex = payload_hex[:96] + "..."
            self._log("Detect OCR parse error: %s payload=%s" % (repr(e), payload_hex))
            return

        try:
            self._emit_detect_result(parsed, raw_values=raw_values)
        except Exception as e:
            self._log("Detect OCR emit error: %s data=%s" % (repr(e), parsed))

    def _handle_detect_color(self, payload):
        """处理颜色检测结果"""
        if not payload:
            return
        try:
            def parse_blobs(data, offset, blob_count, raw_values, rotated=False):
                blobs = []
                for _ in range(blob_count):
                    if offset + 10 > len(data):
                        raise ValueError("blob payload too short")
                    coord0, coord1, w, h = struct.unpack_from('>HHHH', data, offset)
                    offset += 8
                    angle = struct.unpack_from('>h', data, offset)[0]
                    offset += 2
                    raw_values.extend((coord0, coord1, w, h, angle))
                    if rotated:
                        blobs.append({
                            'geometry': 'rotated_rect',
                            'cx': coord0,
                            'cy': coord1,
                            'w': w,
                            'h': h,
                            'angle': angle,
                        })
                    else:
                        blobs.append({'x': coord0, 'y': coord1, 'w': w, 'h': h, 'angle': angle})
                return blobs, offset

            def parse_grouped_rotated(data):
                offset = 0
                if offset >= len(data):
                    raise ValueError("grouped rotated color missing color_count")
                color_count = data[offset]
                offset += 1
                raw_values = [color_count]
                results = []
                for _ in range(color_count):
                    color_name, consumed, declared_len = self._decode_raw_string_values(data, offset)
                    offset += consumed
                    raw_values.append(declared_len)
                    raw_values.append(color_name)
                    if offset >= len(data):
                        raise ValueError("grouped rotated color missing blob_count")
                    blob_count = data[offset]
                    offset += 1
                    raw_values.append(blob_count)
                    blobs, offset = parse_blobs(data, offset, blob_count, raw_values, rotated=True)
                    results.append({'color': color_name, 'blobs': blobs})
                if offset != len(data):
                    raise ValueError("grouped rotated color trailing bytes")
                return {'type': 'multi_color', 'results': results}, raw_values

            parsed = None
            raw_values = None
            try:
                parsed, raw_values = parse_grouped_rotated(payload)
            except Exception:
                parsed = None
                raw_values = None
                raise

            self._emit_detect_result(parsed, raw_values=raw_values)
        except Exception as e:
            self._log("Detect color parse error: %s" % e)

    def _handle_detect_line(self, payload):
        """处理线检测结果"""
        if not payload:
            return
        try:
            def parse_blob_list(data, offset, blob_count, raw_values):
                blobs = []
                for _ in range(blob_count):
                    blob_size = 9
                    if offset + blob_size > len(data):
                        raise ValueError("line blob payload too short")
                    index = data[offset]
                    offset += 1
                    raw_values.append(index)
                    x, y, w, h = struct.unpack_from('>HHHH', data, offset)
                    offset += 8
                    raw_values.extend((x, y, w, h))
                    blob = {'index': index, 'x': x, 'y': y, 'w': w, 'h': h}
                    blobs.append(blob)
                return blobs, offset

            def parse_current(data):
                offset = 0
                if offset >= len(data):
                    raise ValueError("line payload missing blob_count")
                blob_count = data[offset]
                offset += 1
                raw_values = []
                raw_values.append(blob_count)
                color_name, consumed, declared_len = self._decode_raw_string_values(data, offset)
                offset += consumed
                raw_values.append(declared_len)
                raw_values.append(color_name)
                if offset + 4 > len(data):
                    raise ValueError("line payload missing center/angle")
                center_pos = struct.unpack_from('>h', data, offset)[0]
                offset += 2
                angle = struct.unpack_from('>h', data, offset)[0]
                offset += 2
                raw_values.extend((center_pos, angle))
                blobs, offset = parse_blob_list(data, offset, blob_count, raw_values)
                if offset != len(data):
                    raise ValueError("line payload trailing bytes")
                return color_name, blobs, center_pos, angle, raw_values

            color_name, blobs, center_pos, angle, raw_values = parse_current(payload)

            self._emit_detect_result({
                'type': 'line', 'color': color_name, 'blobs': blobs,
                'center_pos': center_pos, 'angle': angle
            }, raw_values=raw_values)
        except Exception as e:
            self._log("Detect line parse error: %s" % e)

    def _handle_detect_keypoint(self, payload):
        """处理人体关键点结果"""
        if not payload:
            return
        try:
            def parse_keypoint_payload(with_score):
                offset = 0
                count = payload[offset]
                offset += 1
                raw_values = [count]
                results = []

                for _ in range(count):
                    if offset + 68 > len(payload):
                        raise ValueError("keypoint payload truncated before points")
                    keypoints = []
                    for _ in range(34):
                        x = struct.unpack_from('>h', payload, offset)[0]
                        offset += 2
                        keypoints.append(x)
                        raw_values.append(x)
                    name, consumed, declared_len = self._decode_raw_string_values(payload, offset)
                    if consumed <= 0 or offset + consumed > len(payload):
                        raise ValueError("keypoint payload missing name")
                    offset += consumed
                    raw_values.extend((declared_len, name))
                    item = {
                        'keypoints': keypoints,
                        'id_len': declared_len,
                        'id': name,
                        'name_len': declared_len,
                        'name': name,
                    }
                    if with_score:
                        if offset + 2 > len(payload):
                            raise ValueError("keypoint payload missing score")
                        score = struct.unpack_from('>h', payload, offset)[0]
                        offset += 2
                        raw_values.append(score)
                        item['score'] = score
                    results.append(item)

                if offset != len(payload):
                    raise ValueError("keypoint payload trailing bytes")
                return {'type': 'keypoint', 'results': results}, raw_values

            last_error = None
            for with_score in (True, False):
                try:
                    parsed, raw_values = parse_keypoint_payload(with_score)
                    self._emit_detect_result(parsed, raw_values=raw_values)
                    return
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
            raise ValueError("keypoint payload parse failed")
        except Exception as e:
            self._log("Detect keypoint parse error: %s" % e)

    def _handle_detect_hand_kp(self, payload):
        """处理手部关键点结果"""
        if not payload:
            return
        try:
            def parse_hand_kp_payload(with_score):
                offset = 0
                count = payload[offset]
                offset += 1
                raw_values = [count]
                results = []

                for _ in range(count):
                    if offset + 92 > len(payload):
                        raise ValueError("hand keypoint payload truncated before points")
                    x, y, w, h = struct.unpack_from('>HHHH', payload, offset)
                    offset += 8
                    raw_values.extend((x, y, w, h))
                    keypoints = []
                    for _ in range(42):
                        value = struct.unpack_from('>h', payload, offset)[0]
                        keypoints.append(value)
                        raw_values.append(value)
                        offset += 2
                    name, consumed, declared_len = self._decode_raw_string_values(payload, offset)
                    if consumed <= 0 or offset + consumed > len(payload):
                        raise ValueError("hand keypoint payload missing name")
                    offset += consumed
                    raw_values.extend((declared_len, name))

                    item = {
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'keypoints': keypoints,
                        'id_len': declared_len,
                        'id': name,
                        'name_len': declared_len,
                        'name': name,
                    }
                    if with_score:
                        if offset + 2 > len(payload):
                            raise ValueError("hand keypoint payload missing score")
                        score = struct.unpack_from('>h', payload, offset)[0]
                        offset += 2
                        raw_values.append(score)
                        item['score'] = score
                    results.append(item)

                if offset != len(payload):
                    raise ValueError("hand keypoint payload trailing bytes")
                return {'type': 'hand_kp', 'results': results}, raw_values

            last_error = None
            for with_score in (True, False):
                try:
                    parsed, raw_values = parse_hand_kp_payload(with_score)
                    self._emit_detect_result(parsed, raw_values=raw_values)
                    return
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
            raise ValueError("hand keypoint payload parse failed")
        except Exception as e:
            self._log("Detect hand keypoint parse error: %s" % e)

    def _handle_detect_center(self, payload):
        """处理精简中心检测结果"""
        if not payload:
            return
        try:
            offset = 0
            count = payload[offset]
            offset += 1
            raw_values = [count]

            results = []
            for _ in range(count):
                if offset + 4 > len(payload):
                    break
                x = struct.unpack_from('>H', payload, offset)[0]
                offset += 2
                y = struct.unpack_from('>H', payload, offset)[0]
                offset += 2
                raw_values.extend((x, y))

                extra = []
                if offset < len(payload):
                    extra_count = payload[offset]
                    offset += 1
                    raw_values.append(extra_count)
                    for _ in range(extra_count):
                        if offset >= len(payload):
                            break
                        extra_type = payload[offset]
                        offset += 1
                        raw_values.append(extra_type)
                        if extra_type == 1:  # 字符串
                            s, consumed, declared_len = self._decode_raw_string_values(payload, offset)
                            extra.append(s)
                            raw_values.append(declared_len)
                            raw_values.append(s)
                            offset += consumed
                        elif extra_type == 2:  # 整数
                            if offset + 2 <= len(payload):
                                val = struct.unpack_from('>h', payload, offset)[0]
                                extra.append(val)
                                raw_values.append(val)
                                offset += 2
                        elif extra_type == 3:  # 整数(乘100)
                            if offset + 2 <= len(payload):
                                val = struct.unpack_from('>h', payload, offset)[0]
                                converted = val / 100.0
                                extra.append(converted)
                                raw_values.append(converted)
                                offset += 2

                results.append({'x': x, 'y': y, 'extra': extra})

            self._emit_detect_result({'type': 'center', 'results': results}, raw_values=raw_values)
        except Exception as e:
            self._log("Detect center parse error: %s" % e)

    def _handle_detect_face_kp(self, payload):
        """处理人脸关键点结果"""
        if not payload:
            return
        try:
            offset = 0
            count = payload[offset]
            offset += 1
            raw_values = [count]

            results = []
            for _ in range(count):
                if offset + 20 > len(payload):
                    break
                keypoints = []
                for _ in range(10):
                    kp = struct.unpack_from('>h', payload, offset)[0]
                    offset += 2
                    keypoints.append(kp)
                    raw_values.append(kp)
                results.append({'keypoints': keypoints})

            self._emit_detect_result({'type': 'face_kp', 'results': results}, raw_values=raw_values)
        except Exception as e:
            self._log("Detect face keypoint parse error: %s" % e)
