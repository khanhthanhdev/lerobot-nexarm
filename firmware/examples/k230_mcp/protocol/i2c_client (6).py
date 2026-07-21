# -*- coding: utf-8 -*-
"""
K230 I2C Master Communication Client
(K230 I2C 主机通信客户端)

Based on protocol_common.py I2C master communication client.
(基于 protocol_common.py 的 I2C 主机通信客户端)
Used for K230 MicroPython environment as I2C master to communicate with slave.
(用于 K230 MicroPython 环境作为 I2C 主机与从机通信)

I2C Mailbox v2 (I2C mailbox v2 说明):
========================================
1. Shared window size: 4096 bytes
   (共享窗口大小: 4096 字节)
   - Mailbox header occupies first 32 bytes
     (前 32 字节为 mailbox 头)
   - Then host slot and device slot data areas
     (之后是主机槽和设备槽的数据区)

2. Slot metadata driven state machine
   (通过 slot 元数据驱动收发状态机)
   - EMPTY -> WRITING -> READY
     (空 -> 正在写 -> 可读)
   - Reader acknowledges by writing slot state back to EMPTY
     (接收方通过把 slot 状态写回 EMPTY 来确认已读)

3. Frame size: configurable via I2C_MAX_FRAME_SIZE, default 256 bytes
   (单帧大小: 通过 I2C_MAX_FRAME_SIZE 配置，默认 256 字节)
   - Max payload = frame_size - 8
     (最大载荷 = frame_size - 8)

Fragmented transfer (分片传输):
========================================
- Payload larger than current slot frame size is automatically fragmented
  (超过当前单帧上限时自动分片)
- Reassembly checks frame type, func, txn and sequence continuity
  (重组时校验帧类型、功能码、事务号和序号连续性)
"""

import time
import struct
from machine import I2C, FPIOA

from protocol_common import (
    K230ClientBase, FrameParser, PayloadReassembler, HeartbeatData,
    FRAME_HEADER_0, FRAME_HEADER_1, FRAME_TYPE_CMD, FRAME_TYPE_RPT,
    FRAME_LEN_OFFSET, FRAME_FUNC_OFFSET, FRAME_PAYLOAD_OFFSET,
    MIN_FRAME_LEN, MAX_PAYLOAD_LEN, MAX_FRAME_LEN, SEQ_MASK,
    CMD_REQUEST_STATUS, CMD_CLEAR_MEMORY, CMD_SET_WIFI, CMD_ASR,
    CMD_EMPTY_RETURN, CMD_RESULT_RETURN, CMD_GET_PROTOCOL_INFO,
    RPT_HEARTBEAT, RPT_ERROR, RPT_DETECT_BBOX, RPT_DETECT_STR, RPT_DETECT_OCR,
    RPT_DETECT_COLOR, RPT_DETECT_LINE, RPT_DETECT_KEYPOINT, RPT_DETECT_HAND_KP,
    RPT_DETECT_CENTER, RPT_DETECT_FACE_KP, RPT_DETECT_QUAD, CMD_ASR_TIMEOUT_MS,
    CompactCodec,
    build_frame, split_payload, calc_xor,
    APP_INDEX_TO_NAME, ERROR_NAMES, ERROR_MODULE_NAMES, ERR_OK
)

# I2C Configuration (I2C 配置)
I2C_DEV_ADDR = 0x5F
I2C_MEM_SIZE = 4096               # Mailbox shared window size, default 4096, max 4096 (mailbox 共享窗口大小，默认4096，最大4096)
I2C_BUS = 2
I2C_MAX_FRAME_SIZE = 256  # Max frame size, default 256 (mailbox 单帧最大大小，默认256)
I2C_MAX_PAYLOAD = min(MAX_PAYLOAD_LEN, I2C_MAX_FRAME_SIZE - MIN_FRAME_LEN)  # 单帧最大载荷
I2C_MAILBOX_READY_TIMEOUT_MS = 5000
I2C_STATUS_COOLDOWN_MS = 8  # 请求状态命令冷却时间，过大会把接收帧率压低
I2C_STATUS_WAIT_STEP_MS = 2  # 轮询等待从机回包的步进
I2C_STATUS_WAIT_TIMEOUT_MS = 80  # 单次状态请求的最长期待回包时间
I2C_CMD_ECHO_RETRY_COUNT = 3
I2C_CMD_ECHO_RETRY_STEP_MS = 2
I2C_MAIN_LOOP_INTERVAL_MS = 5
I2C_DETECT_LOG_INTERVAL_MS = 200
I2C_STATS_LOG_INTERVAL_MS = 1000
I2C_DUP_READY_STALL_LOG_MS = 200
I2C_DUP_READY_STALL_LOG_INTERVAL_MS = 1000
_TEST_RESULT_UNSET = object()
_TEST_INSTRUCTION_KEY = '__instruction__'
_TEST_INSTRUCTION_TEXT = {
    'asr_ready': '可以开始说话了',
}
_MCP_AUTO_REPLY_IGNORE_KEYS = (
    _TEST_INSTRUCTION_KEY,
    'asr',
    'llm',
    'vlm',
    'mcp',
    'tool',
    'ok',
    'result',
)


def _make_test_instruction(name):
    return {_TEST_INSTRUCTION_KEY: name}


def _format_value_for_log(value):
    if isinstance(value, dict):
        instruction_name = value.get(_TEST_INSTRUCTION_KEY)
        if isinstance(instruction_name, str):
            return repr(_TEST_INSTRUCTION_TEXT.get(instruction_name, instruction_name))
    if isinstance(value, bytes):
        if not value:
            return "b''"
        try:
            return repr(value.decode('utf-8'))
        except Exception:
            return value.hex()
    return repr(value)


def _format_detect_result_for_log(result):
    def _preferred_keys(value):
        if not isinstance(value, dict):
            return ()
        result_type = value.get('type')
        if result_type == 'multi_color':
            return ('results', 'type')
        if result_type == 'color':
            return ('color', 'blobs', 'type')
        if result_type == 'line':
            return ('color', 'blobs', 'center_pos', 'angle', 'type')
        if result_type in ('bbox', 'keypoint', 'hand_kp', 'center', 'face_kp', 'ocr', 'quad', 'string_list'):
            return ('results', 'type')
        if result_type == 'string':
            return ('value', 'type')
        if 'color' in value and 'blobs' in value:
            return ('color', 'blobs')
        if value.get('geometry') == 'rotated_rect':
            return ('cx', 'cy', 'w', 'h', 'angle', 'geometry')
        if 'x' in value and 'y' in value and 'w' in value and 'h' in value:
            return ('x', 'y', 'w', 'h', 'angle', 'geometry', 'extra')
        if 'points' in value:
            return ('points', 'text', 'extra')
        return ()

    def _format_any(value):
        if isinstance(value, dict):
            preferred = []
            seen = set()
            for key in _preferred_keys(value):
                if key in value and key not in seen:
                    preferred.append(key)
                    seen.add(key)
            remaining = []
            for key in value.keys():
                if key not in seen:
                    remaining.append(key)
                    seen.add(key)
            parts = []
            for key in preferred + remaining:
                parts.append("%r: %s" % (key, _format_any(value[key])))
            return '{' + ', '.join(parts) + '}'
        if isinstance(value, list):
            return '[' + ', '.join(_format_any(item) for item in value) + ']'
        return repr(value)

    return _format_any(result)


def _format_rate_text(sample_count, elapsed_ms):
    if sample_count <= 0 or elapsed_ms <= 0:
        return "0.0"

    rate_x10 = int((sample_count * 10000 + (elapsed_ms // 2)) // elapsed_ms)
    return "%d.%d" % (rate_x10 // 10, rate_x10 % 10)


def _detect_result_summary_key(result):
    if not isinstance(result, dict):
        return ('unknown', 0)

    result_type = result.get('type', '')
    count = 0
    if result_type == 'string':
        value = result.get('value')
        count = 1 if isinstance(value, str) and value.strip() else 0
    else:
        results = result.get('results')
        if isinstance(results, list):
            count = len(results)
        elif results is not None:
            count = 1 if results else 0

    return (result_type, count)


def _extract_mcp_tool_call(result):
    if not isinstance(result, dict) or len(result) != 1:
        return None

    tool_name, tool_args = next(iter(result.items()))
    if not isinstance(tool_name, str) or not tool_name:
        return None
    if tool_name in _MCP_AUTO_REPLY_IGNORE_KEYS:
        return None
    return tool_name, tool_args


def _build_mock_mcp_reply(tool_name, tool_args):
    if tool_name == 'run_action':
        action_name = ''
        if isinstance(tool_args, dict):
            raw_name = tool_args.get('name', '')
            if raw_name is not None:
                action_name = str(raw_name).strip()
        return "模拟执行动作成功: %s" % (action_name or 'unknown')

    return "模拟执行成功: %s" % tool_name


def _make_i2c_mcp_result_callback(client, pending_state, auto_reply=True):
    def _callback(result):
        client._log("MCP工具结果: %s" % _format_value_for_log(result))

        if not auto_reply:
            return

        tool_call = _extract_mcp_tool_call(result)
        if tool_call is None:
            return

        tool_name, tool_args = tool_call
        reply_text = _build_mock_mcp_reply(tool_name, tool_args)
        pending_state.setdefault('queue', []).append({
            'tool_name': tool_name,
            'reply_text': reply_text,
            'next_retry_at': None,
            'last_fail_log_ms': None,
        })
        client._log(
            "检测到MCP工具调用，准备自动模拟返回: tool=%s, result=%s" % (
                tool_name,
                _format_value_for_log(reply_text),
            )
        )

    return _callback


def _flush_pending_mcp_replies(client, pending_state):
    queue = pending_state.get('queue')
    if not queue:
        return

    item = queue[0]
    now = time.ticks_ms()
    next_retry_at = item.get('next_retry_at')
    if next_retry_at is not None and time.ticks_diff(now, next_retry_at) < 0:
        return

    tool_name = item.get('tool_name', '')
    reply_text = item.get('reply_text', '')

    try:
        ok = bool(client.send_mcp_result(reply_text))
    except Exception as exc:
        ok = False
        err_text = str(exc)
    else:
        err_text = None

    if ok:
        queue.pop(0)
        client._log(
            "MCP模拟返回已发送: tool=%s, result=%s" % (
                tool_name,
                _format_value_for_log(reply_text),
            )
        )
        return

    item['next_retry_at'] = time.ticks_add(now, 500)
    last_fail_log_ms = item.get('last_fail_log_ms')
    if (
        last_fail_log_ms is None
        or time.ticks_diff(now, last_fail_log_ms) >= 1000
    ):
        if err_text:
            client._log("MCP模拟返回发送异常，稍后重试: tool=%s, err=%s" % (tool_name, err_text))
        else:
            client._log("MCP模拟返回发送失败，稍后重试: tool=%s" % tool_name)
        item['last_fail_log_ms'] = now


class I2CClient(K230ClientBase):
    """K230 I2C 主机通信客户端（mailbox v2）"""

    MAILBOX_MAGIC = b'WLM2'
    MAILBOX_HEADER_SIZE = 32
    SLOT_META_SIZE = 8
    HOST_SLOT_META_OFFSET = 16
    DEVICE_SLOT_META_OFFSET = 24
    SLOT_DATA_OFFSET = 32
    SLOT_STATE_EMPTY = 0
    SLOT_STATE_WRITING = 1
    SLOT_STATE_READY = 2
    SLOT_HOST = 0
    SLOT_DEVICE = 1

    def __init__(self, bus=I2C_BUS, addr=I2C_DEV_ADDR, scl_pin=11, sda_pin=12,
                 mem_size=I2C_MEM_SIZE, fragment_enabled=True):
        super().__init__()

        self.bus = bus
        self.addr = addr
        self.scl_pin = scl_pin
        self.sda_pin = sda_pin
        self.mem_size = mem_size
        self.fragment_enabled = fragment_enabled
        self.i2c = None
        self.slot_size = I2C_MAX_FRAME_SIZE
        self.max_frame_size = I2C_MAX_FRAME_SIZE
        self.max_payload_size = I2C_MAX_PAYLOAD
        self._host_generation = 0
        self._last_device_generation = 0
        self._log_start_ms = time.ticks_ms()
        self._error_count = 0
        self._max_errors = 10
        self.detect_log_interval_ms = I2C_DETECT_LOG_INTERVAL_MS
        self.stats_log_interval_ms = I2C_STATS_LOG_INTERVAL_MS
        self.trace_runtime_debug = False
        self._stuck_device_generation = None
        self._stuck_device_ready_since_ms = None
        self._last_stuck_device_log_ms = None

        self.reassembler.set_time_func(lambda: time.ticks_ms())
        self._init_i2c()

    def _log(self, msg):
        ticks_now = time.ticks_ms()
        mono_ms = time.ticks_diff(ticks_now, self._log_start_ms)
        mono_stamp = "%d.%03d" % (mono_ms // 1000, mono_ms % 1000)
        print("[%s][I2C] %s" % (mono_stamp, msg))

    def _is_ide_interrupt(self, error):
        error_str = str(error)
        return "IDE interrupt" in error_str or isinstance(error, KeyboardInterrupt)

    def _handle_i2c_error(self, error):
        if self._is_ide_interrupt(error):
            raise error
        self._error_count += 1
        if self._error_count >= self._max_errors:
            self._log("Too many I2C errors, reinitializing...")
            self._reinit_i2c()
            self._error_count = 0

    def _init_i2c(self):
        try:
            fpioa = FPIOA()
            iic_funcs = [
                (FPIOA.IIC0_SCL, FPIOA.IIC0_SDA),
                (FPIOA.IIC1_SCL, FPIOA.IIC1_SDA),
                (FPIOA.IIC2_SCL, FPIOA.IIC2_SDA),
                (FPIOA.IIC3_SCL, FPIOA.IIC3_SDA),
                (FPIOA.IIC4_SCL, FPIOA.IIC4_SDA),
            ]
            if 0 <= self.bus <= 4:
                scl_func, sda_func = iic_funcs[self.bus]
                fpioa.set_function(self.scl_pin, scl_func)
                fpioa.set_function(self.sda_pin, sda_func)

            self.i2c = I2C(self.bus, freq=400000)
            self._detect_device()
            if self._wait_mailbox_ready(I2C_MAILBOX_READY_TIMEOUT_MS):
                self._log("Mailbox v2 ready")
                self._recover_stale_host_slot('init')
            else:
                self._log("Mailbox v2 not ready")
            self._error_count = 0
            self._log("I2C Master initialized on bus %d, addr=0x%02X" % (self.bus, self.addr))
            return True
        except Exception as e:
            self._log("I2C init failed: %s" % e)
            return False

    def _reinit_i2c(self):
        try:
            self.i2c = None
            time.sleep_ms(100)
            return self._init_i2c()
        except Exception as e:
            self._log("I2C reinit failed: %s" % e)
            return False

    def _detect_device(self):
        try:
            devices = self.i2c.scan()
            if self.addr in devices:
                self._log("I2C device found at 0x%02X" % self.addr)
            else:
                self._log("I2C device NOT found at 0x%02X" % self.addr)
        except Exception as e:
            self._log("I2C scan error: %s" % e)

    def _read_mem(self, offset, length):
        try:
            return self.i2c.readfrom_mem(self.addr, offset, length, addrsize=16)
        except TypeError:
            try:
                return self.i2c.readfrom_mem(self.addr, offset, length, addr_size=16)
            except TypeError:
                return self.i2c.readfrom_mem(self.addr, offset, length, 16)

    def _write_mem(self, offset, data):
        try:
            self.i2c.writeto_mem(self.addr, offset, data, addrsize=16)
        except TypeError:
            try:
                self.i2c.writeto_mem(self.addr, offset, data, addr_size=16)
            except TypeError:
                self.i2c.writeto_mem(self.addr, offset, data, 16)

    def _slot_meta_offset(self, slot):
        return self.DEVICE_SLOT_META_OFFSET if slot == self.SLOT_DEVICE else self.HOST_SLOT_META_OFFSET

    def _slot_data_offset(self, slot):
        return self.SLOT_DATA_OFFSET + (self.slot_size if slot == self.SLOT_DEVICE else 0)

    def _read_slot_meta(self, slot):
        raw = self._read_mem(self._slot_meta_offset(slot), self.SLOT_META_SIZE)
        return {
            'state': raw[0],
            'generation': struct.unpack('>H', raw[2:4])[0],
            'frame_len': struct.unpack('>H', raw[4:6])[0],
            'frame_xor': raw[6],
        }

    def _write_slot_meta(self, slot, state, generation, frame_len=0, frame_xor=0):
        raw = bytearray(self.SLOT_META_SIZE)
        raw[0] = state & 0xFF
        raw[2:4] = struct.pack('>H', generation & 0xFFFF)
        raw[4:6] = struct.pack('>H', frame_len & 0xFFFF)
        raw[6] = frame_xor & 0xFF
        self._write_mem(self._slot_meta_offset(slot), raw)

    def _wait_slot_empty(self, slot, timeout_ms=500):
        start = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            try:
                if self._read_slot_meta(slot)['state'] == self.SLOT_STATE_EMPTY:
                    return True
            except Exception:
                pass
            time.sleep_ms(2)
        return False

    def _describe_slot(self, slot):
        try:
            meta = self._read_slot_meta(slot)
            return "state=%d gen=%d len=%d xor=0x%02X" % (
                meta.get('state', -1),
                meta.get('generation', 0),
                meta.get('frame_len', 0),
                meta.get('frame_xor', 0),
            )
        except Exception as exc:
            return "read_error=%s" % exc

    def _recover_stale_host_slot(self, reason):
        try:
            meta = self._read_slot_meta(self.SLOT_HOST)
        except Exception as exc:
            self._log("Recover host slot skipped reason=%s read_error=%s" % (reason, exc))
            return False

        state = meta.get('state', self.SLOT_STATE_EMPTY)
        generation = meta.get('generation', 0)
        if generation > self._host_generation:
            self._host_generation = generation
        if state == self.SLOT_STATE_EMPTY:
            return False

        self._write_slot_meta(self.SLOT_HOST, self.SLOT_STATE_EMPTY, generation)
        self._log(
            "Recover stale host slot reason=%s state=%d gen=%d len=%d" % (
                reason,
                state,
                generation,
                meta.get('frame_len', 0),
            )
        )
        time.sleep_ms(2)
        return True

    def _refresh_mailbox_ready(self):
        try:
            header = self._read_mem(0, 8)
            if header[:4] != self.MAILBOX_MAGIC:
                return False
            slot_size = struct.unpack('>H', header[6:8])[0]
            if slot_size < MIN_FRAME_LEN:
                return False
            self.slot_size = slot_size
            self.max_frame_size = slot_size
            self.max_payload_size = max(1, min(MAX_PAYLOAD_LEN, slot_size - MIN_FRAME_LEN))
            return True
        except Exception:
            return False

    def _wait_mailbox_ready(self, timeout_ms=I2C_MAILBOX_READY_TIMEOUT_MS):
        start = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            if self._refresh_mailbox_ready():
                return True
            time.sleep_ms(10)
        return False

    def _send_frame_to_host_slot(self, frame):
        frame = bytes(frame)
        if len(frame) > self.slot_size:
            raise ValueError("frame too large for host slot")

        self._host_generation = (self._host_generation + 1) & 0xFFFF
        if self._host_generation == 0:
            self._host_generation = 1
        frame_xor = calc_xor(frame)
        self._write_slot_meta(self.SLOT_HOST,
                              self.SLOT_STATE_WRITING,
                              self._host_generation,
                              len(frame),
                              frame_xor)
        self._write_mem(self._slot_data_offset(self.SLOT_HOST), frame)
        self._write_slot_meta(self.SLOT_HOST,
                              self.SLOT_STATE_READY,
                              self._host_generation,
                              len(frame),
                              frame_xor)

    def _ack_device_slot(self, generation):
        self._write_slot_meta(self.SLOT_DEVICE, self.SLOT_STATE_EMPTY, generation)

    def _reset_stuck_device_slot_state(self):
        self._stuck_device_generation = None
        self._stuck_device_ready_since_ms = None
        self._last_stuck_device_log_ms = None

    def _retry_ack_stuck_device_slot(self, meta):
        now = time.ticks_ms()
        generation = meta.get('generation', 0)
        frame_len = meta.get('frame_len', 0)

        if self._stuck_device_generation != generation:
            self._stuck_device_generation = generation
            self._stuck_device_ready_since_ms = now
            self._last_stuck_device_log_ms = None
        else:
            stall_ms = time.ticks_diff(now, self._stuck_device_ready_since_ms)
            need_log = (
                stall_ms >= I2C_DUP_READY_STALL_LOG_MS and
                (
                    self._last_stuck_device_log_ms is None or
                    time.ticks_diff(now, self._last_stuck_device_log_ms) >= I2C_DUP_READY_STALL_LOG_INTERVAL_MS
                )
            )
            if need_log:
                self._log(
                    "Device slot stuck READY: generation=%d, len=%d, retry EMPTY ack" % (
                        generation,
                        frame_len,
                    )
                )
                self._last_stuck_device_log_ms = now

        self._ack_device_slot(generation)

    def send_command(self, func_code, payload=b'', txn_id=None):
        if not self.i2c:
            return False
        if txn_id is None:
            txn_id = self._next_txn()

        try:
            if not self._refresh_mailbox_ready():
                if not self._wait_mailbox_ready(I2C_MAILBOX_READY_TIMEOUT_MS):
                    self._log("Mailbox v2 still not ready, skip cmd=0x%02X" % func_code)
                    return False
            if not self._wait_slot_empty(self.SLOT_HOST, timeout_ms=500):
                recovered = self._recover_stale_host_slot('send_cmd_0x%02X' % func_code)
                if not recovered or not self._wait_slot_empty(self.SLOT_HOST, timeout_ms=50):
                    self._log(
                        "Host slot busy, skip cmd=0x%02X slot=%s" % (
                            func_code,
                            self._describe_slot(self.SLOT_HOST),
                        )
                    )
                    return False

            chunks = split_payload(payload, self.max_payload_size)
            for chunk_data, continuation in chunks:
                frame = build_frame(FRAME_TYPE_CMD,
                                    self._next_seq(),
                                    func_code,
                                    chunk_data,
                                    continuation,
                                    txn_id=txn_id)
                self._send_frame_to_host_slot(frame)
                if continuation and not self._wait_slot_empty(self.SLOT_HOST, timeout_ms=1000):
                    return False

            if self.debug_mode and func_code != CMD_REQUEST_STATUS:
                self._log("[TX] cmd=0x%02X, txn=%d, len=%d" % (func_code, txn_id, len(payload)))
            return True
        except Exception as e:
            self._log("Send command error: %s" % e)
            self._handle_i2c_error(e)
            return False

    def send_command_and_wait(self, func_code, payload=b'', timeout_ms=8000, max_retries=2, txn_id=None):
        if txn_id is None:
            txn_id = self._next_txn()

        for retry in range(max_retries + 1):
            self._cmd_result_data = None
            self._cmd_success = False
            self._pending_cmd = func_code
            self._pending_txn = txn_id

            if not self.send_command(func_code, payload, txn_id=txn_id):
                if retry >= max_retries:
                    self._pending_cmd = None
                    self._pending_txn = None
                    return False, None
                time.sleep_ms(50)
                continue

            start = time.ticks_ms()
            while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
                self.poll_status()
                if self._cmd_result_data is not None:
                    return self._cmd_success, self._cmd_result_data
                time.sleep_ms(5)

            if retry < max_retries:
                self._log("Command 0x%02X timeout, retry %d/%d" % (func_code, retry + 1, max_retries))
                time.sleep_ms(50)

        self._pending_cmd = None
        self._pending_txn = None
        return False, None

    def send_command_and_wait_async_result(self, func_code, payload=b'', timeout_ms=30000, wait_cmd=None):
        if wait_cmd is None:
            wait_cmd = CMD_RESULT_RETURN

        self._async_result = None
        self._async_result_ready = False
        self._async_success = False
        self._waiting_async_cmd = wait_cmd
        self._waiting_async_owner_cmd = func_code
        self._waiting_async_txn = None

        success, _ = self.send_command_and_wait(func_code, payload, timeout_ms=5000)
        if not success:
            self._waiting_async_cmd = None
            self._waiting_async_owner_cmd = None
            self._waiting_async_txn = None
            return False, None

        self._waiting_async_txn = self._last_response_txn
        start = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            self.poll_status()
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
            time.sleep_ms(10)

        self._waiting_async_cmd = None
        self._waiting_async_owner_cmd = None
        self._waiting_async_txn = None
        return False, None

    def _read_and_process_response(self):
        return self._read_and_process_device_frame()

    def _read_and_process_device_frame(self):
        if not self.i2c:
            return False

        try:
            meta = self._read_slot_meta(self.SLOT_DEVICE)
            if meta['state'] != self.SLOT_STATE_READY:
                self._reset_stuck_device_slot_state()
                return False
            if meta['generation'] == self._last_device_generation:
                self._retry_ack_stuck_device_slot(meta)
                return False
            if meta['frame_len'] < MIN_FRAME_LEN or meta['frame_len'] > self.slot_size:
                self._ack_device_slot(meta['generation'])
                self._last_device_generation = meta['generation']
                self._reset_stuck_device_slot_state()
                return False

            raw = self._read_mem(self._slot_data_offset(self.SLOT_DEVICE), meta['frame_len'])
            self._ack_device_slot(meta['generation'])
            self._last_device_generation = meta['generation']
            self._reset_stuck_device_slot_state()

            if calc_xor(raw) != meta['frame_xor']:
                return False

            self.parser.clear()
            self.parser.feed(bytes(raw))
            frames = self.parser.parse_all()
            if not frames:
                return False

            for frame in frames:
                self._handle_frame(frame)
            self._error_count = 0
            return True
        except Exception as e:
            self._handle_i2c_error(e)
            return False

    def request_status(self, timeout_ms=3000):
        """请求设备立即回一次最新状态。

        当前协议 v2 下设备会先返回 CMD_REQUEST_STATUS 的同步 RSP，
        再通过 device slot 主动补发新的状态心跳；这里以
        “收到新的 heartbeat”作为完成判据。
        """
        start_heartbeat_serial = self._heartbeat_serial
        success, result_data = self.send_command_and_wait(CMD_REQUEST_STATUS, b'', timeout_ms=timeout_ms)
        if not success:
            return False, result_data

        start = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            self.poll_status()
            if self._heartbeat_serial != start_heartbeat_serial:
                return True, result_data
            time.sleep_ms(10)
        return False, result_data

    def poll_status(self):
        return self._read_and_process_device_frame()

    @staticmethod
    def is_ide_interrupt(exception):
        return ("IDE interrupt" in str(exception) or
                isinstance(exception, KeyboardInterrupt))

    def start(self):
        self.run()

    def run(self, interval_ms=I2C_MAIN_LOOP_INTERVAL_MS):
        self._log("Starting main loop...")
        self._running = True
        poll_count = 0
        success_count = 0
        fail_count = 0
        last_stats_ms = time.ticks_ms()
        last_success_time = time.ticks_ms()
        try:
            while self._running:
                result = self.poll_status()
                poll_count += 1
                if result:
                    success_count += 1
                    fail_count = 0
                    last_success_time = time.ticks_ms()
                else:
                    fail_count += 1

                if self.debug_mode and self.trace_runtime_debug:
                    now = time.ticks_ms()
                    if time.ticks_diff(now, last_stats_ms) >= 1000:
                        elapsed = time.ticks_diff(now, last_stats_ms)
                        self._log("Poll: total=%d success=%d elapsed=%dms last_report=%s" % (
                            poll_count,
                            success_count,
                            elapsed,
                            self.describe_last_report(),
                        ))
                        last_stats_ms = now
                    if fail_count > 0 and time.ticks_diff(time.ticks_ms(), last_success_time) > 5000:
                        self._log("No progress for %dms" % time.ticks_diff(time.ticks_ms(), last_success_time))
                time.sleep_ms(interval_ms)
        except Exception as e:
            if self.is_ide_interrupt(e):
                self._log("IDE interrupt detected, exiting...")
            else:
                self._log("Run error: %s" % e)
        finally:
            self._running = False

    def stop(self):
        self._running = False
        self._log("Stopped.")

    def close(self):
        self.deinit()

    def deinit(self):
        self.stop()
        self.i2c = None
        self.parser.clear()
        self.reassembler.reset()
        self._log("I2C Client deinitialized.")


def _i2c_ensure_test_ready(self):
    return


def _i2c_format_test_value(self, value):
    return _format_value_for_log(value)


def _i2c_decode_command_result_bytes(self, result_data=None):
    data = self._cmd_result_data if result_data is None else result_data
    info = {
        'raw': data,
        'cmd': None,
        'code': None,
        'code_name': None,
        'module': None,
        'module_name': None,
        'subcode': None,
        'success': None,
        'extra': None,
    }
    if not data or len(data) < 5:
        return info

    info['cmd'] = data[0]
    info['code'] = data[1]
    info['code_name'] = ERROR_NAMES.get(data[1], 'Unknown(%d)' % data[1])
    info['module'] = data[2]
    info['module_name'] = ERROR_MODULE_NAMES.get(data[2], 'unknown')
    info['subcode'] = ((data[3] << 8) | data[4]) & 0xFFFF
    info['success'] = (data[1] == ERR_OK)
    info['extra'] = data[5:] if len(data) > 5 else None
    return info


def _i2c_normalize_wifi_connected(self, value):
    if isinstance(value, dict):
        return value.get('connected')
    return value


def _i2c_command_result_text(self, info):
    if info.get('cmd') is None:
        return '未收到设备命令回包'

    text = 'cmd=0x%02X, code=0x%02X(%s)' % (
        info['cmd'],
        info['code'],
        info['code_name'],
    )
    if info.get('module') is not None:
        text += ', module=%s(0x%02X)' % (info['module_name'], info['module'])
    if info.get('subcode') is not None:
        text += ', sub=0x%04X' % info['subcode']
    if info.get('extra') is not None:
        text += ', extra=%s' % self._format_test_value(info['extra'])
    return text


def _i2c_print_test_return(self, test_name, actual_result):
    print('[RETURN] %s: %s' % (test_name, self._format_test_value(actual_result)))


def _i2c_finish_test(self,
                     test_name,
                     problems,
                     actual_result=_TEST_RESULT_UNSET,
                     result_data=None):
    summary = {
        'test': test_name,
        'ok': len(problems) == 0,
        'result': None if actual_result is _TEST_RESULT_UNSET else actual_result,
        'problems': list(problems),
        'command_result': self._decode_command_result_bytes(result_data),
    }

    if problems:
        if actual_result is not _TEST_RESULT_UNSET:
            self._print_test_return(test_name, actual_result)
        for problem in problems:
            print('[ISSUE] %s: %s' % (test_name, problem))
    elif actual_result is _TEST_RESULT_UNSET:
        print('[OK] %s' % test_name)
    else:
        print('[OK] %s: %s' % (test_name, self._format_test_value(actual_result)))
    return summary


def _i2c_run_success_test(self, test_name, action, expected_success=True):
    self._ensure_test_ready()
    self._cmd_result_data = None
    problems = []

    try:
        actual_success = bool(action())
    except Exception as exc:
        problems.append('执行异常: %s' % exc)
        return self._finish_test(test_name, problems)

    info = self._decode_command_result_bytes()
    if actual_success != expected_success:
        problems.append(
            '成功状态不符合预期: expected=%s, actual=%s, %s' % (
                expected_success,
                actual_success,
                self._command_result_text(info),
            )
        )
    if expected_success and not actual_success and info.get('cmd') is None:
        problems.append('命令执行失败且未收到设备回包')

    return self._finish_test(test_name, problems, result_data=info.get('raw'))


def _i2c_run_tuple_result_test(self,
                               test_name,
                               action,
                               expected_success=True,
                               expected_result=_TEST_RESULT_UNSET,
                               normalizer=None):
    self._ensure_test_ready()
    self._cmd_result_data = None
    problems = []

    try:
        actual_success, actual_result = action()
    except Exception as exc:
        problems.append('执行异常: %s' % exc)
        return self._finish_test(test_name, problems)

    info = self._decode_command_result_bytes()
    if actual_success != expected_success:
        problems.append(
            '成功状态不符合预期: expected=%s, actual=%s, %s' % (
                expected_success,
                actual_success,
                self._command_result_text(info),
            )
        )
    if expected_success and not actual_success and info.get('cmd') is None:
        problems.append('命令执行失败且未收到设备回包')

    if expected_result is not _TEST_RESULT_UNSET:
        actual_value = normalizer(actual_result) if normalizer else actual_result
        expected_value = normalizer(expected_result) if normalizer else expected_result
        if actual_value != expected_value:
            problems.append(
                '返回值不符合预期: expected=%s, actual=%s' % (
                    self._format_test_value(expected_result),
                    self._format_test_value(actual_result),
                )
            )

    return self._finish_test(
        test_name,
        problems,
        actual_result=actual_result,
        result_data=info.get('raw'),
    )


def _i2c_run_value_result_test(self,
                               test_name,
                               action,
                               expected_success=True,
                               expected_result=_TEST_RESULT_UNSET,
                               normalizer=None):
    self._ensure_test_ready()
    self._cmd_result_data = None
    problems = []

    try:
        actual_result = action()
    except Exception as exc:
        problems.append('执行异常: %s' % exc)
        return self._finish_test(test_name, problems)

    info = self._decode_command_result_bytes()
    actual_success = info['success'] if info['success'] is not None else (actual_result is not None)
    if actual_success != expected_success:
        problems.append(
            '成功状态不符合预期: expected=%s, actual=%s, %s' % (
                expected_success,
                actual_success,
                self._command_result_text(info),
            )
        )
    if expected_success and not actual_success and info.get('cmd') is None:
        problems.append('命令执行失败且未收到设备回包')

    if expected_result is not _TEST_RESULT_UNSET:
        actual_value = normalizer(actual_result) if normalizer else actual_result
        expected_value = normalizer(expected_result) if normalizer else expected_result
        if actual_value != expected_value:
            problems.append(
                '返回值不符合预期: expected=%s, actual=%s' % (
                    self._format_test_value(expected_result),
                    self._format_test_value(actual_result),
                )
            )

    return self._finish_test(
        test_name,
        problems,
        actual_result=actual_result,
        result_data=info.get('raw'),
    )


def _i2c_wait_async_result(self, timeout_ms):
    start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
        self._read_and_process_response()
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
        time.sleep_ms(10)

    self._waiting_async_cmd = None
    self._waiting_async_owner_cmd = None
    self._waiting_async_txn = None
    return None


def _i2c_test_set_wifi(self,
                       ssid,
                       password,
                       expected_success=True,
                       expected_connected=_TEST_RESULT_UNSET,
                       timeout_ms=30000):
    def action():
        payload = CompactCodec.encode_string(ssid) + CompactCodec.encode_string(password)
        success, result = self.send_command_and_wait_async_result(
            CMD_SET_WIFI,
            payload,
            wait_cmd=CMD_EMPTY_RETURN,
            timeout_ms=timeout_ms,
        )
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

    return self._run_tuple_result_test(
        'CMD_SET_WIFI',
        action,
        expected_success=expected_success,
        expected_result=expected_connected,
        normalizer=self._normalize_wifi_connected,
    )


def _i2c_test_request_status(self, expected_success=True, timeout_ms=3000):
    self._ensure_test_ready()
    self._cmd_result_data = None
    problems = []
    result_data = None

    try:
        actual_success, result_data = self.request_status(timeout_ms=timeout_ms)
    except Exception as exc:
        problems.append('执行异常: %s' % exc)
        return self._finish_test('CMD_REQUEST_STATUS', problems)

    if actual_success != expected_success:
        problems.append(
            '成功状态不符合预期: expected=%s, actual=%s' % (
                expected_success,
                actual_success,
            )
        )
    if expected_success and not actual_success:
        problems.append('未收到状态上报')

    actual_result = {
        'heartbeat': self.get_heartbeat(),
    }
    return self._finish_test('CMD_REQUEST_STATUS',
                             problems,
                             actual_result=actual_result,
                             result_data=result_data)


def _i2c_test_start_asr(self,
                        start=True,
                        expected_success=True,
                        expected_result=_TEST_RESULT_UNSET):
    if not start:
        return self._run_tuple_result_test(
            'CMD_ASR',
            lambda: self.start_asr(start),
            expected_success=expected_success,
            expected_result=expected_result,
        )

    self._ensure_test_ready()
    self._cmd_result_data = None
    problems = []
    actual_success = False
    actual_result = None

    try:
        payload = CompactCodec.encode_uint8(1)
        self._async_result = None
        self._async_result_ready = False
        self._async_success = False
        self._waiting_async_cmd = CMD_RESULT_RETURN
        self._waiting_async_owner_cmd = CMD_ASR

        actual_success, _ = self.send_command_and_wait(CMD_ASR, payload, timeout_ms=5000)
        if actual_success:
            self._print_test_return('CMD_ASR', _make_test_instruction('asr_ready'))
            async_result = self._wait_async_result(CMD_ASR_TIMEOUT_MS)
            if async_result is None:
                actual_success = False
                actual_result = None
            else:
                actual_success, actual_result = async_result
        else:
            self._waiting_async_cmd = None
            self._waiting_async_owner_cmd = None
    except Exception as exc:
        problems.append('执行异常: %s' % exc)
        return self._finish_test('CMD_ASR', problems)

    info = self._decode_command_result_bytes()
    if actual_success != expected_success:
        problems.append(
            '成功状态不符合预期: expected=%s, actual=%s, %s' % (
                expected_success,
                actual_success,
                self._command_result_text(info),
            )
        )
    if expected_success and not actual_success and info.get('cmd') is None:
        problems.append('命令执行失败且未收到设备回包')

    if expected_result is not _TEST_RESULT_UNSET and actual_result != expected_result:
        problems.append(
            '返回值不符合预期: expected=%s, actual=%s' % (
                self._format_test_value(expected_result),
                self._format_test_value(actual_result),
            )
        )

    return self._finish_test(
        'CMD_ASR',
        problems,
        actual_result=actual_result,
        result_data=info.get('raw'),
    )


def _make_i2c_success_test_method(command_name, target_method_name):
    def method(self, *args, **kwargs):
        expected_success = kwargs.pop('expected_success', True)
        return self._run_success_test(
            command_name,
            lambda: getattr(self, target_method_name)(*args, **kwargs),
            expected_success,
        )

    try:
        method.__doc__ = '测试 %s。' % command_name
    except (AttributeError, TypeError):
        pass
    return method


def _make_i2c_value_test_method(command_name, target_method_name, normalizer=None):
    def method(self, *args, **kwargs):
        expected_success = kwargs.pop('expected_success', True)
        expected_result = kwargs.pop('expected_result', _TEST_RESULT_UNSET)
        return self._run_value_result_test(
            command_name,
            lambda: getattr(self, target_method_name)(*args, **kwargs),
            expected_success=expected_success,
            expected_result=expected_result,
            normalizer=normalizer,
        )

    try:
        method.__doc__ = '测试 %s。' % command_name
    except (AttributeError, TypeError):
        pass
    return method


def _make_i2c_tuple_test_method(command_name, target_method_name, normalizer=None):
    def method(self, *args, **kwargs):
        expected_success = kwargs.pop('expected_success', True)
        expected_result = kwargs.pop('expected_result', _TEST_RESULT_UNSET)
        return self._run_tuple_result_test(
            command_name,
            lambda: getattr(self, target_method_name)(*args, **kwargs),
            expected_success=expected_success,
            expected_result=expected_result,
            normalizer=normalizer,
        )

    try:
        method.__doc__ = '测试 %s。' % command_name
    except (AttributeError, TypeError):
        pass
    return method


def _install_i2c_test_methods():
    helper_methods = {
        '_ensure_test_ready': _i2c_ensure_test_ready,
        '_format_test_value': _i2c_format_test_value,
        '_decode_command_result_bytes': _i2c_decode_command_result_bytes,
        '_normalize_wifi_connected': _i2c_normalize_wifi_connected,
        '_command_result_text': _i2c_command_result_text,
        '_print_test_return': _i2c_print_test_return,
        '_finish_test': _i2c_finish_test,
        '_run_success_test': _i2c_run_success_test,
        '_run_tuple_result_test': _i2c_run_tuple_result_test,
        '_run_value_result_test': _i2c_run_value_result_test,
        '_wait_async_result': _i2c_wait_async_result,
    }
    for name, func in helper_methods.items():
        setattr(I2CClient, name, func)

    success_specs = (
        ('test_set_mode', 'CMD_SET_MODE', 'set_mode'),
        ('test_set_volume', 'CMD_SET_VOLUME', 'set_volume'),
        ('test_clear_memory', 'CMD_CLEAR_MEMORY', 'clear_memory'),
        ('test_set_confidence_threshold', 'CMD_SET_CONF_THRESH', 'set_confidence_threshold'),
        ('test_set_nms_threshold', 'CMD_SET_NMS_THRESH', 'set_nms_threshold'),
        ('test_set_mask_threshold', 'CMD_SEG_SET_MASK_THRESH', 'set_mask_threshold'),
        ('test_set_simple_result_mode', 'CMD_SET_SIMPLE_RESULT', 'set_simple_result_mode'),
        ('test_set_run_enabled', 'CMD_DISABLE_RUN', 'set_run_enabled'),
        ('test_face_learn', 'CMD_FACE_LEARN', 'face_learn'),
        ('test_face_pose_learn', 'CMD_FACE_LEARN(mode=FacePose)', 'face_pose_learn'),
        ('test_eye_gaze_learn', 'CMD_FACE_LEARN(mode=EyeGaze)', 'eye_gaze_learn'),
        ('test_face_enhance_learn', 'CMD_FACE_ENHANCE_LEARN', 'face_enhance_learn'),
        ('test_face_pose_enhance_learn', 'CMD_FACE_ENHANCE_LEARN(mode=FacePose)', 'face_pose_enhance_learn'),
        ('test_eye_gaze_enhance_learn', 'CMD_FACE_ENHANCE_LEARN(mode=EyeGaze)', 'eye_gaze_enhance_learn'),
        ('test_face_learn_at_point', 'CMD_FACE_LEARN_AT_POINT', 'face_learn_at_point'),
        ('test_face_pose_learn_at_point', 'CMD_FACE_LEARN_AT_POINT(mode=FacePose)', 'face_pose_learn_at_point'),
        ('test_eye_gaze_learn_at_point', 'CMD_FACE_LEARN_AT_POINT(mode=EyeGaze)', 'eye_gaze_learn_at_point'),
        ('test_face_delete', 'CMD_FACE_DELETE', 'face_delete'),
        ('test_face_pose_delete', 'CMD_FACE_DELETE(mode=FacePose)', 'face_pose_delete'),
        ('test_eye_gaze_delete', 'CMD_FACE_DELETE(mode=EyeGaze)', 'eye_gaze_delete'),
        ('test_face_rename', 'CMD_FACE_RENAME', 'face_rename'),
        ('test_face_pose_rename', 'CMD_FACE_RENAME(mode=FacePose)', 'face_pose_rename'),
        ('test_eye_gaze_rename', 'CMD_FACE_RENAME(mode=EyeGaze)', 'eye_gaze_rename'),
        ('test_set_face_recognition_threshold', 'CMD_FACE_SET_RECOG_CONF', 'set_face_recognition_threshold'),
        ('test_set_face_high_precision', 'CMD_FACE_HIGH_PRECISION', 'set_face_high_precision'),
        ('test_set_face_keypoint_mode', 'CMD_FACE_ENABLE_KEYPOINT', 'set_face_keypoint_mode'),
        ('test_set_face_detect_only_mode', 'CMD_FACE_DETECT_ONLY', 'set_face_detect_only_mode'),
        ('test_person_kp_learn', 'CMD_PERSON_KP_LEARN', 'person_kp_learn'),
        ('test_person_kp_enhance_learn', 'CMD_PERSON_KP_ENHANCE_LEARN', 'person_kp_enhance_learn'),
        ('test_person_kp_delete', 'CMD_PERSON_KP_DELETE', 'person_kp_delete'),
        ('test_person_kp_rename', 'CMD_PERSON_KP_RENAME', 'person_kp_rename'),
        ('test_hand_kp_learn', 'CMD_HAND_KP_LEARN', 'hand_kp_learn'),
        ('test_hand_kp_enhance_learn', 'CMD_HAND_KP_ENHANCE_LEARN', 'hand_kp_enhance_learn'),
        ('test_hand_kp_delete', 'CMD_HAND_KP_DELETE', 'hand_kp_delete'),
        ('test_hand_kp_rename', 'CMD_HAND_KP_RENAME', 'hand_kp_rename'),
        ('test_set_hand_detect_only_mode', 'CMD_HAND_DETECT_ONLY', 'set_hand_detect_only_mode'),
        ('test_set_color_target', 'CMD_COLOR_SET_TARGET', 'set_color_target'),
        ('test_set_color_threshold', 'CMD_COLOR_SET_THRESH', 'set_color_threshold'),
        ('test_set_color_filter', 'CMD_COLOR_SET_FILTER', 'set_color_filter'),
        ('test_set_color_min_area', 'CMD_COLOR_SET_MIN_AREA', 'set_color_min_area'),
        ('test_set_multi_color_list', 'CMD_MULTI_COLOR_SET_LIST', 'set_multi_color_list'),
        ('test_set_line_roi', 'CMD_LINE_SET_ROI', 'set_line_roi'),
        ('test_set_color_learning_point', 'CMD_COLOR_LEARNING_SET_POINT', 'set_color_learning_point'),
        ('test_save_color_learning', 'CMD_COLOR_LEARNING_SAVE', 'save_color_learning'),
        ('test_rename_color_learning', 'CMD_COLOR_LEARNING_RENAME', 'rename_color_learning'),
        ('test_delete_color_learning', 'CMD_COLOR_LEARNING_DELETE', 'delete_color_learning'),
        ('test_selflearn_set_name', 'CMD_SELFLEARN_SET_NAME', 'selflearn_set_name'),
        ('test_selflearn_set_rect', 'CMD_SELFLEARN_SET_RECT', 'selflearn_set_rect'),
        ('test_selflearn_set_frame', 'CMD_SELFLEARN_SET_FRAME', 'selflearn_set_frame'),
        ('test_selflearn_set_features', 'CMD_SELFLEARN_SET_FEATURES', 'selflearn_set_features'),
        ('test_selflearn_delete', 'CMD_SELFLEARN_DELETE', 'selflearn_delete'),
        ('test_selflearn_rename', 'CMD_SELFLEARN_RENAME', 'selflearn_rename'),
        ('test_nanotrack_set_rect', 'CMD_NANOTRACK_SET_RECT', 'nanotrack_set_rect'),
        ('test_nanotrack_stop', 'CMD_NANOTRACK_STOP', 'nanotrack_stop'),
        ('test_set_gesture_frame_count', 'CMD_GESTURE_SET_FRAME', 'set_gesture_frame_count'),
        ('test_dgesture_record_start', 'CMD_DGESTURE_CTRL(RECORD_START)', 'dgesture_record_start'),
        ('test_dgesture_record_stop', 'CMD_DGESTURE_CTRL(RECORD_STOP)', 'dgesture_record_stop'),
        ('test_dgesture_save', 'CMD_DGESTURE_CTRL(SAVE)', 'dgesture_save'),
        ('test_dgesture_enhance_save', 'CMD_DGESTURE_ENHANCE_SAVE', 'dgesture_enhance_save'),
        ('test_dgesture_enhance_save_drop_oldest', 'CMD_DGESTURE_CTRL(SAVE_APPEND_DROP_OLDEST)', 'dgesture_enhance_save_drop_oldest'),
        ('test_dgesture_delete', 'CMD_DGESTURE_CTRL(DELETE)', 'dgesture_delete'),
        ('test_dgesture_rename', 'CMD_DGESTURE_CTRL(RENAME)', 'dgesture_rename'),
        ('test_set_custom_model', 'CMD_CUSTOM_SET_MODEL', 'set_custom_model'),
        ('test_set_object_mode', 'CMD_OBJECT_SET_MODE', 'set_object_mode'),
        ('test_set_face_pose_threshold', 'CMD_FACE_SET_POSE_THRESH', 'set_face_pose_threshold'),
        ('test_set_eye_gaze_threshold', 'CMD_FACE_SET_POSE_THRESH(mode=EyeGaze)', 'set_eye_gaze_threshold'),
        ('test_media_camera_snapshot', 'CMD_MEDIA_CAMERA_SNAPSHOT', 'media_camera_snapshot'),
        ('test_media_set_photo_prefix', 'CMD_MEDIA_SET_PHOTO_PREFIX', 'media_set_photo_prefix'),
        ('test_media_delete_photo', 'CMD_MEDIA_DELETE_PHOTO', 'media_delete_photo'),
        ('test_media_enter_camera_app', 'CMD_MEDIA_ENTER_CAMERA_APP', 'media_enter_camera_app'),
        ('test_media_set_photo_start', 'CMD_MEDIA_SET_PHOTO_START', 'media_set_photo_start'),
        ('test_set_llm_key', 'CMD_SET_LLM_KEY', 'set_llm_key'),
        ('test_set_llm_model', 'CMD_SET_LLM_MODEL', 'set_llm_model'),
        ('test_set_vlm_model', 'CMD_SET_VLM_MODEL', 'set_vlm_model'),
        ('test_set_llm_base_url', 'CMD_SET_LLM_BASE_URL', 'set_llm_base_url'),
        ('test_set_vlm_base_url', 'CMD_SET_VLM_BASE_URL', 'set_vlm_base_url'),
        ('test_set_speech_url', 'CMD_SET_SPEECH_URL', 'set_speech_url'),
        ('test_set_tts_voice', 'CMD_SET_TTS_VOICE', 'set_tts_voice'),
        ('test_set_asr_language', 'CMD_SET_ASR_LANG', 'set_asr_language'),
        ('test_set_thinking_mode', 'CMD_SET_THINKING', 'set_thinking_mode'),
        ('test_set_search_mode', 'CMD_SET_SEARCH', 'set_search_mode'),
        ('test_set_start_silence', 'CMD_SET_START_SILENCE', 'set_start_silence'),
        ('test_set_end_silence', 'CMD_SET_END_SILENCE', 'set_end_silence'),
        ('test_set_system_prompt', 'CMD_SET_PROMPT', 'set_system_prompt'),
        ('test_set_mcp_tools', 'CMD_SET_MCP_TOOLS', 'set_mcp_tools'),
        ('test_send_mcp_result', 'CMD_RESULT_RETURN', 'send_mcp_result'),
    )
    for method_name, command_name, target_method_name in success_specs:
        method = _make_i2c_success_test_method(command_name, target_method_name)
        try:
            method.__name__ = method_name
        except (AttributeError, TypeError):
            pass
        setattr(I2CClient, method_name, method)

    tuple_specs = (
        ('test_tts_speak', 'CMD_TTS', 'tts_speak'),
        ('test_llm_chat', 'CMD_LLM_CHAT', 'llm_chat'),
        ('test_vlm_chat', 'CMD_VLM_CHAT', 'vlm_chat'),
    )
    for method_name, command_name, target_method_name in tuple_specs:
        method = _make_i2c_tuple_test_method(command_name, target_method_name)
        try:
            method.__name__ = method_name
        except (AttributeError, TypeError):
            pass
        setattr(I2CClient, method_name, method)

    I2CClient.test_set_wifi = _i2c_test_set_wifi
    I2CClient.test_request_status = _i2c_test_request_status
    I2CClient.test_start_asr = _i2c_test_start_asr

    def test_get_protocol_info(self,
                               expected_success=True,
                               expected_result=_TEST_RESULT_UNSET):
        return self._run_value_result_test(
            'CMD_GET_PROTOCOL_INFO',
            self.get_protocol_info,
            expected_success=expected_success,
            expected_result=expected_result,
        )

    I2CClient.test_get_protocol_info = test_get_protocol_info

    def test_get_color_threshold(self,
                                 color_name,
                                 expected_threshold=_TEST_RESULT_UNSET,
                                 expected_success=True):
        return self._run_value_result_test(
            'CMD_COLOR_GET_THRESH',
            lambda: self.get_color_threshold(color_name),
            expected_success=expected_success,
            expected_result=expected_threshold,
        )

    I2CClient.test_get_color_threshold = test_get_color_threshold

    def test_select_color_learning_profile(self, name, expected_success=True):
        return self._run_success_test(
            'CMD_COLOR_SET_TARGET',
            lambda: self.select_color_learning_profile(name),
            expected_success=expected_success,
        )

    I2CClient.test_select_color_learning_profile = test_select_color_learning_profile


_install_i2c_test_methods()


def _detect_result_has_targets(result):
    """判断当前检测结果是否包含可用于后续注册/保存的有效目标。"""
    result_type = result.get('type', '')

    if result_type == 'string':
        value = result.get('value')
        return isinstance(value, str) and bool(value.strip())

    results = result.get('results')
    if isinstance(results, list):
        if result_type == 'string_list':
            for item in results:
                if isinstance(item, str) and item.strip():
                    return True
            return False
        return len(results) > 0

    return bool(results)


def _make_detect_trigger_action(mode_name,
                                subject_name,
                                action_desc,
                                result_types,
                                action,
                                hold_ms=2000,
                                gap_ms=1000):
    """构造“检测稳定后执行一次动作”的配置。"""
    return {
        'mode_name': mode_name,
        'subject_name': subject_name,
        'action_desc': action_desc,
        'result_types': result_types,
        'action': action,
        'hold_ms': hold_ms,
        'gap_ms': gap_ms,
    }


def _make_post_enable_action(action_desc, action, delay_ms=500):
    """构造“模式启用后延后执行一次动作”的配置。"""
    return {
        'action_desc': action_desc,
        'action': action,
        'delay_ms': max(0, delay_ms),
    }


def _run_example_loop(client,
                      post_enable_actions,
                      detect_trigger_action,
                      detect_trigger_state,
                      mcp_reply_state=None,
                      trace_state=None,
                      interval_ms=I2C_MAIN_LOOP_INTERVAL_MS):
    """示例主循环。

    用于对齐 UART 客户端中的两类时序：
    1. detect_trigger_action: 依赖检测稳定一段时间后再执行一次
    2. post_enable_actions: 依赖模式已 enable/run，延后一次执行
    """
    client._running = True
    next_post_enable_at = None
    if post_enable_actions:
        next_post_enable_at = time.ticks_add(time.ticks_ms(), post_enable_actions[0]['delay_ms'])

    try:
        while client._running:
            last_rx_frame_ms_before_poll = client.last_rx_frame_ms
            poll_ok = client.poll_status()
            now = time.ticks_ms()
            if trace_state is not None:
                trace_state['stats_loop_count'] = trace_state.get('stats_loop_count', 0) + 1
                if poll_ok and trace_state.get('last_silence_log_ms') is not None:
                    if last_rx_frame_ms_before_poll is None:
                        gap_ms = time.ticks_diff(now, trace_state.get('startup_ms', now))
                    else:
                        gap_ms = time.ticks_diff(now, last_rx_frame_ms_before_poll)
                    client._log(
                        "通信恢复: kind=transport, gap_ms=%d, last_frame=%s" % (
                            gap_ms,
                            client.describe_last_frame(),
                        )
                    )
                    trace_state['last_silence_log_ms'] = None

            if mcp_reply_state is not None:
                _flush_pending_mcp_replies(client, mcp_reply_state)

            if trace_state is not None:
                last_transport_ms = client.last_rx_frame_ms
                last_silence_log_ms = trace_state.get('last_silence_log_ms')
                if (
                    ((last_transport_ms is None and time.ticks_diff(now, trace_state.get('startup_ms', now)) >= 3000) or
                     (last_transport_ms is not None and time.ticks_diff(now, last_transport_ms) >= 3000))
                    and (
                        last_silence_log_ms is None
                        or time.ticks_diff(now, last_silence_log_ms) >= 3000
                    )
                ):
                    last_heartbeat_ms = trace_state.get('last_heartbeat_ms')
                    last_detect_ms = trace_state.get('last_detect_ms')
                    heartbeat_gap = '--'
                    detect_gap = '--'
                    if last_heartbeat_ms is not None:
                        heartbeat_gap = str(time.ticks_diff(now, last_heartbeat_ms))
                    if last_detect_ms is not None:
                        detect_gap = str(time.ticks_diff(now, last_detect_ms))
                    report_gap = '--'
                    response_gap = '--'
                    frame_gap = '--'
                    no_transport_ms = time.ticks_diff(now, trace_state.get('startup_ms', now))
                    if client.last_rx_frame_ms is not None:
                        frame_gap = str(time.ticks_diff(now, client.last_rx_frame_ms))
                        no_transport_ms = time.ticks_diff(now, client.last_rx_frame_ms)
                    if client.last_report_ms is not None:
                        report_gap = str(time.ticks_diff(now, client.last_report_ms))
                    if client.last_response_ms is not None:
                        response_gap = str(time.ticks_diff(now, client.last_response_ms))
                    client._log(
                        "通信空窗: no_transport_ms=%d, last_frame_gap_ms=%s, last_report_gap_ms=%s, last_rsp_gap_ms=%s, last_detect_gap_ms=%s, last_heartbeat_gap_ms=%s, last_frame=%s, last_report=%s, last_rsp=%s" % (
                            no_transport_ms,
                            frame_gap,
                            report_gap,
                            response_gap,
                            detect_gap,
                            heartbeat_gap,
                            client.describe_last_frame(),
                            client.describe_last_report(),
                            client.describe_last_response(),
                        )
                    )
                    trace_state['last_silence_log_ms'] = now

                stats_window_start_ms = trace_state.get('stats_window_start_ms')
                if stats_window_start_ms is not None:
                    stats_elapsed_ms = time.ticks_diff(now, stats_window_start_ms)
                    stats_interval_ms = max(200, int(getattr(client, 'stats_log_interval_ms', I2C_STATS_LOG_INTERVAL_MS)))
                    if stats_elapsed_ms >= stats_interval_ms:
                        last_detect_key = trace_state.get('last_detect_summary_key')
                        last_detect_text = '--'
                        if last_detect_key is not None:
                            last_detect_text = "%s:%d" % (last_detect_key[0], last_detect_key[1])
                        rx_frame_count = client.rx_frame_serial - trace_state.get('stats_window_rx_frame_serial', client.rx_frame_serial)
                        report_count = client.report_serial - trace_state.get('stats_window_report_serial', client.report_serial)
                        response_count = client.response_serial - trace_state.get('stats_window_response_serial', client.response_serial)
                        client._log(
                            "主机统计: loop=%sHz, rx_frame=%sHz, report=%sHz, rsp=%sHz, detect=%sHz, heartbeat=%sHz, last_frame=%s, last_report=%s, last_rsp=%s, last_detect=%s" % (
                                _format_rate_text(trace_state.get('stats_loop_count', 0), stats_elapsed_ms),
                                _format_rate_text(rx_frame_count, stats_elapsed_ms),
                                _format_rate_text(report_count, stats_elapsed_ms),
                                _format_rate_text(response_count, stats_elapsed_ms),
                                _format_rate_text(trace_state.get('stats_detect_count', 0), stats_elapsed_ms),
                                _format_rate_text(trace_state.get('stats_heartbeat_count', 0), stats_elapsed_ms),
                                client.describe_last_frame(),
                                client.describe_last_report(),
                                client.describe_last_response(),
                                last_detect_text,
                            )
                        )
                        trace_state['stats_window_start_ms'] = now
                        trace_state['stats_window_rx_frame_serial'] = client.rx_frame_serial
                        trace_state['stats_window_report_serial'] = client.report_serial
                        trace_state['stats_window_response_serial'] = client.response_serial
                        trace_state['stats_loop_count'] = 0
                        trace_state['stats_detect_count'] = 0
                        trace_state['stats_heartbeat_count'] = 0

            if (
                post_enable_actions
                and next_post_enable_at is not None
                and time.ticks_diff(now, next_post_enable_at) >= 0
            ):
                action_cfg = post_enable_actions.pop(0)
                client._log("模式已启用，执行一次: %s" % action_cfg['action_desc'])
                action_result = action_cfg['action']()
                if isinstance(action_result, dict):
                    action_ok = action_result.get('ok')
                else:
                    action_ok = action_result
                client._log("延后操作结果: ok=%s" % action_ok)
                next_post_enable_at = None
                if post_enable_actions:
                    next_post_enable_at = time.ticks_add(now, post_enable_actions[0]['delay_ms'])

            if detect_trigger_action:
                last_detected_at = detect_trigger_state['last_detected_at']
                stable_since = detect_trigger_state['stable_since']
                gap_ms = detect_trigger_action['gap_ms']
                hold_ms = detect_trigger_action['hold_ms']

                if (
                    last_detected_at is not None
                    and time.ticks_diff(now, last_detected_at) > gap_ms
                ):
                    detect_trigger_state['last_detected_at'] = None
                    detect_trigger_state['stable_since'] = None
                    stable_since = None

                if (
                    stable_since is not None
                    and not detect_trigger_state['attempted']
                    and time.ticks_diff(now, stable_since) >= hold_ms
                ):
                    client._log(
                        "检测到%s已持续 %.1f 秒，执行一次: %s" % (
                            detect_trigger_action['subject_name'],
                            hold_ms / 1000.0,
                            detect_trigger_action['action_desc'],
                        )
                    )
                    action_result = detect_trigger_action['action']()
                    if isinstance(action_result, dict):
                        action_ok = action_result.get('ok')
                    else:
                        action_ok = action_result
                    detect_trigger_state['attempted'] = True
                    client._log("自动操作结果: ok=%s" % action_ok)

            time.sleep_ms(interval_ms)
    finally:
        client._running = False


# ============================================================================
# 程序入口点
# ============================================================================
def main():
    client = None
    try:
        client = I2CClient(bus=2, addr=0x5F, scl_pin=11, sda_pin=12)
        client.debug_mode = True
        client.show_heartbeat = True
        client.trace_raw_i2c = False  # 高频轮询默认关闭原始 I2C 追踪，避免日志本身拖慢接收

        post_enable_actions = []
        trace_state = {
            'startup_ms': time.ticks_ms(),
            'last_detect_ms': None,
            'last_heartbeat_ms': None,
            'last_silence_log_ms': None,
            'last_heartbeat_key': None,
            'last_detect_log_ms': None,
            'last_detect_log_key': None,
            'last_heartbeat_log_ms': None,
            'last_detect_summary_key': None,
            'stats_window_start_ms': time.ticks_ms(),
            'stats_window_rx_frame_serial': client.rx_frame_serial,
            'stats_window_report_serial': client.report_serial,
            'stats_window_response_serial': client.response_serial,
            'stats_loop_count': 0,
            'stats_detect_count': 0,
            'stats_heartbeat_count': 0,
        }
        mcp_reply_state = {
            'queue': [],
        }
        detect_trigger_state = {
            'attempted': False,
            'stable_since': None,
            'last_detected_at': None,
        }
        detect_trigger_action = None

        single_color_threshold = {
            'red': [0, 100, -64, 64, -64, 64],
        }
        multi_color_threshold = {
            'red': [0, 100, -64, 64, -64, 64],
            'green': [0, 100, -64, 64, -64, 64],
        }
        line_color_threshold = {
            'black': [0, 40, -20, 20, -20, 20],
        }
        line_roi = [
            [0, 0, 320, 40, 10],
            [0, 40, 320, 40, 10],
            [0, 80, 320, 40, 10],
        ]
        mcp_tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_action",
                    "description": "用户要求执行某个动作时可以调用这个工具, 可选动作有:'sit_dowm':坐下, 'go_prone':趴下，",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "用户要求执行的动作名称",
                            }
                        },
                        "required": ["name"],
                    },
                },
                "block": 5,
            },
        ]
        mcp_result = {
            'tool': 'time_now',
            'ok': True,
            'result': '2026-03-30 12:00:00',
        }

        def on_detect(result):
            now = time.ticks_ms()
            trace_state['last_detect_ms'] = now
            summary_key = _detect_result_summary_key(result)
            trace_state['stats_detect_count'] = trace_state.get('stats_detect_count', 0) + 1
            trace_state['last_detect_summary_key'] = summary_key

            last_detect_log_ms = trace_state.get('last_detect_log_ms')
            last_detect_log_key = trace_state.get('last_detect_log_key')
            detect_log_interval_ms = max(20, int(getattr(client, 'detect_log_interval_ms', I2C_DETECT_LOG_INTERVAL_MS)))
            if (
                last_detect_log_ms is None
                or summary_key != last_detect_log_key
                or time.ticks_diff(now, last_detect_log_ms) >= detect_log_interval_ms
            ):
                client._log("检测结果: %s" % _format_detect_result_for_log(result))
                trace_state['last_detect_log_ms'] = now
                trace_state['last_detect_log_key'] = summary_key

            if not detect_trigger_action or detect_trigger_state['attempted']:
                return

            if client.current_mode_name != detect_trigger_action['mode_name']:
                return

            result_type = result.get('type', '')
            if result_type not in detect_trigger_action['result_types']:
                return

            if _detect_result_has_targets(result):
                last_detected_at = detect_trigger_state['last_detected_at']
                gap_ms = detect_trigger_action['gap_ms']
                if (
                    last_detected_at is None
                    or time.ticks_diff(now, last_detected_at) > gap_ms
                ):
                    detect_trigger_state['stable_since'] = now
                detect_trigger_state['last_detected_at'] = now

        def on_heartbeat(heartbeat):
            trace_state['last_heartbeat_ms'] = time.ticks_ms()
            trace_state['stats_heartbeat_count'] = trace_state.get('stats_heartbeat_count', 0) + 1

            if not client.show_heartbeat:
                return

            status = heartbeat.get('status', {})
            heartbeat_key = (
                heartbeat.get('mode'),
                bool(status.get('run')),
                bool(status.get('ready')),
                bool(status.get('busy')),
                bool(status.get('result')),
                bool(status.get('error')),
            )
            idle_detect_ms = None
            if trace_state['last_detect_ms'] is not None:
                idle_detect_ms = time.ticks_diff(time.ticks_ms(), trace_state['last_detect_ms'])

            should_log = heartbeat_key != trace_state['last_heartbeat_key']
            if not should_log and idle_detect_ms is not None and idle_detect_ms >= 3000:
                last_heartbeat_log_ms = trace_state.get('last_heartbeat_log_ms')
                should_log = (
                    last_heartbeat_log_ms is None or
                    time.ticks_diff(time.ticks_ms(), last_heartbeat_log_ms) >= 3000
                )

            if should_log:
                extra = ""
                if idle_detect_ms is not None and idle_detect_ms >= 3000:
                    extra = ", idle_detect_ms=%d" % idle_detect_ms
                client._log(
                    "心跳摘要: mode=%s, run=%s, ready=%s, busy=%s, result=%s, error=%s%s" % (
                        heartbeat.get('mode_name'),
                        bool(status.get('run')),
                        bool(status.get('ready')),
                        bool(status.get('busy')),
                        bool(status.get('result')),
                        bool(status.get('error')),
                        extra,
                    )
                )
                trace_state['last_heartbeat_log_ms'] = time.ticks_ms()
            trace_state['last_heartbeat_key'] = heartbeat_key

        def on_command_result(result):
            if result.get('success'):
                return
            client._log(
                "命令失败: cmd=0x%02X, code=0x%02X(%s), module=%s(0x%02X), sub=0x%04X, extra=%s" % (
                    result['cmd'],
                    result.get('code', 0),
                    result.get('code_name'),
                    result.get('module_name', 'unknown'),
                    result.get('module', 0),
                    result.get('subcode', 0),
                    result.get('extra'),
                )
            )

        def on_llm_result(result):
            if _extract_mcp_tool_call(result) is not None:
                return
            client._log("LLM结果: %s" % result)

        client.on_detect_result = on_detect
        client.on_heartbeat = on_heartbeat
        client.on_command_result = on_command_result
        client.on_mcp_result = _make_i2c_mcp_result_callback(client, mcp_reply_state, auto_reply=True)
        client.on_llm_result = on_llm_result

        # 系统级示例指令，与具体模式无关，可按需取消注释。
        # 完整覆盖清单见 docs/wonderlens_system_host_test_commands.md
        # client.test_clear_memory()
        # client.test_set_volume(70)
        # client.test_set_wifi('WWW', 'hiwonder', expected_connected=True)
        # client.test_request_status()
        # time.sleep_ms(1000)
        client.test_set_simple_result_mode(False)
        # client.test_set_run_enabled(False)
        auto_enable_run = True
        # 可选模式:
        #   视觉模式: Empty / FaceRecognition / ... / AiLLM_Mode
        #   媒体示例: MediaEnterCameraApp / MediaSetPhotoPrefix / MediaSetPhotoStart
        #            MediaCameraSnapshot
        #            MediaDeletePhoto
        mode = 'FaceDetection'
        if mode == 'Empty':
            client.test_set_mode(mode)
            # Empty 模式只切模式，不包含专用 test_* 指令。
        elif mode == 'FaceDetection':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'FaceLandmark':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'FaceMesh':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'FaceRecognition':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
            # client.test_set_face_recognition_threshold(90)
            client.test_set_face_high_precision(False)
            # client.test_set_face_high_precision(True)
            client.test_set_face_keypoint_mode(False)
            # client.test_set_face_keypoint_mode(True)
            client.test_set_face_detect_only_mode(False)
            # client.test_set_face_detect_only_mode(True)
            client.test_set_simple_result_mode(False)
            # 加强注册命令自身支持“若名称不存在则新建，若已存在则追加样本”，
            # 因此这里统一走一条自动动作即可。
            detect_trigger_action = _make_detect_trigger_action(
                mode_name=mode,
                subject_name='人脸',
                action_desc='自动注册/加强注册: face_a',
                result_types=('bbox', 'center', 'face_kp'),
                action=lambda: client.test_face_enhance_learn('甲乙人'),
            )
            # client.test_face_learn_at_point(160, 120, 'face_a')
            # client.test_face_rename('face_a', 'face_b')
            # client.test_face_delete('face_b')
        elif mode == 'FacePose':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
            # FacePose 模式支持通用置信度/NMS 阈值控制。
        elif mode == 'FaceParse':
            client.test_set_mode(mode)
            # client.test_set_confidence_threshold(100)
            # client.test_set_nms_threshold(45)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
            # FaceParse 模式支持通用置信度/NMS 阈值控制。
        elif mode == 'FaceLiveness':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'EyeGaze':
            client.test_set_mode(mode)
            # client.test_set_confidence_threshold(70)
            # client.test_set_nms_threshold(100)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
            # EyeGaze 模式支持通用置信度/NMS 阈值控制。
        elif mode == 'PersonDetection':
            client.test_set_mode(mode)
            # client.test_set_confidence_threshold(60)
            # client.test_set_nms_threshold(100)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'PersonKeypointDetect':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
            # 加强注册命令自身支持“若名称不存在则新建，若已存在则追加样本”，
            # 因此这里统一走一条自动动作即可。
            detect_trigger_action = _make_detect_trigger_action(
                mode_name=mode,
                subject_name='人体关键点',
                action_desc='自动注册/加强注册: person_a',
                result_types=('keypoint',),
                action=lambda: client.test_person_kp_enhance_learn('person_a'),
            )
            # client.test_person_kp_rename('person_a', 'person_b')
            # client.test_person_kp_delete('person_b')
        elif mode == 'HandDetection':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
            post_enable_actions.append(_make_post_enable_action(
                action_desc='设置手掌 detect_only=1',
                action=lambda: client.test_set_hand_detect_only_mode(True),
            ))
        elif mode == 'HandRecognition':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'HandKeyPointDetection':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
            post_enable_actions.append(_make_post_enable_action(
                action_desc='设置手掌关键点 detect_only=0',
                action=lambda: client.test_set_hand_detect_only_mode(False),
            ))
            # post_enable_actions.append(_make_post_enable_action(
            #     action_desc='设置手掌关键点 detect_only=1',
            #     action=lambda: client.test_set_hand_detect_only_mode(True),
            # ))
            # 加强注册命令自身支持“若名称不存在则新建，若已存在则追加样本”，
            # 因此这里统一走一条自动动作即可。
            detect_trigger_action = _make_detect_trigger_action(
                mode_name=mode,
                subject_name='手掌关键点',
                action_desc='自动注册/加强注册: hand_a',
                result_types=('bbox', 'center', 'hand_kp'),
                action=lambda: client.test_hand_kp_enhance_learn('hand_a'),
            )
            # client.test_hand_kp_rename('hand_a', 'hand_b')
            # client.test_hand_kp_delete('hand_b')
        elif mode == 'HandGesture':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'FalldownDetection':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'SingleColorDetection':
            client.test_set_mode(mode)
            client.test_set_color_target('blue')
            # client.test_set_color_threshold(single_color_threshold)
            client.test_get_color_threshold('blue')
            # client.test_get_color_threshold('red', expected_threshold=[0, 100, -64, 64, -64, 64])
            client.test_set_color_filter(1, 100, 50000)
            client.test_set_color_min_area(200)
        elif mode == 'MultiColorDetection':
            client.test_set_mode(mode)
            # client.test_set_color_threshold(multi_color_threshold)
            client.test_set_multi_color_list(['red', 'green'])
            # client.test_set_color_filter(2, 100, 50000)
            # client.test_set_color_min_area(150)
            client.test_get_color_threshold('red')
        elif mode == 'LineDetection':
            client.test_set_mode(mode)
            client.test_set_color_target('black')
            # client.test_set_color_threshold(line_color_threshold)
            # client.test_set_line_roi(line_roi)
            # client.test_set_color_filter(1, 50, 50000)
            # client.test_set_color_min_area(50)
            # client.test_get_color_threshold('black')
        elif mode == 'ColorTracking':
            client.test_set_mode(mode)
            # client.test_set_color_threshold({'sample_a': [0, 100, -64, 64, -64, 64]})
            # post_enable_actions.append(_make_post_enable_action(
            #     action_desc='颜色学习取色一次: ID1 @ (160,120)',
            #     action=lambda: client.test_set_color_learning_point(160, 120, 'ID1'),
            #     delay_ms=800,
            # ))
            # post_enable_actions.append(_make_post_enable_action(
            #     action_desc='保存颜色学习颜色项: ID1',
            #     action=lambda: client.test_save_color_learning('ID1'),
            # ))
            # client.test_rename_color_learning('ID1', 'ID2')
            # client.test_get_color_threshold('ID2')
            # client.test_delete_color_learning('ID2')
            # client.test_select_color_learning_profile('red')
        elif mode == 'SelfLearning':
            client.test_set_mode(mode)
            post_enable_actions.extend([
                # _make_post_enable_action(
                #     action_desc='设置自学习框选区域',
                #     action=lambda: client.test_selflearn_set_rect(60, 40, 120, 120),
                #     delay_ms=600,
                # ),
                # _make_post_enable_action(
                #     action_desc='设置自学习采样帧数=20',
                #     action=lambda: client.test_selflearn_set_frame(20),
                #     delay_ms=200,
                # ),
                # _make_post_enable_action(
                #     action_desc='设置自学习特征数=5',
                #     action=lambda: client.test_selflearn_set_features(5),
                #     delay_ms=200,
                # ),
                # _make_post_enable_action(
                #     action_desc='开始自学习采样: obj_a',
                #     action=lambda: client.test_selflearn_set_name('ID1'),
                #     delay_ms=200,
                # ),
            ])
            client.test_selflearn_rename('ID2', 'ID1')
            # client.test_selflearn_delete('ID1')
        elif mode == 'ObjectTrack':
            client.test_set_mode(mode)
            post_enable_actions.append(_make_post_enable_action(
                action_desc='目标跟踪框选一次: (100, 100, 80, 80)',
                action=lambda: client.test_nanotrack_set_rect(100, 100, 80, 80),
                delay_ms=800,
            ))
            # post_enable_actions.append(_make_post_enable_action(
            #     action_desc='停止目标跟踪',
            #     action=lambda: client.test_nanotrack_stop(True),
            # ))
        elif mode == 'DynamicGesture':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
            post_enable_actions.extend([
                _make_post_enable_action(
                    action_desc='设置动态手势采样帧数=8',
                    action=lambda: client.test_set_gesture_frame_count(8),
                ),
                # _make_post_enable_action(
                #     action_desc='开始动态手势录制',
                #     action=lambda: client.test_dgesture_record_start(),
                #     delay_ms=800,
                # ),
                # _make_post_enable_action(
                #     action_desc='停止动态手势录制',
                #     action=lambda: client.test_dgesture_record_stop(),
                #     delay_ms=2500,
                # ),
                # _make_post_enable_action(
                #     action_desc='保存动态手势: gesture_a',
                #     action=lambda: client.test_dgesture_save('gesture_a'),
                #     delay_ms=300,
                # ),
            ])
            # client.test_dgesture_enhance_save('gesture_a')
            # client.test_dgesture_enhance_save_drop_oldest('gesture_a')
            # client.test_dgesture_rename('gesture_a', 'gesture_b')
            # client.test_dgesture_delete('gesture_b')
        elif mode == 'OCRDetection':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'OCRRecognition':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'LicencePlateDetection':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'LicencePlateRecognition':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'ObjectDetection':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'Segmentation':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
            client.test_set_mask_threshold(50)
            client.test_set_object_mode('seg')
        elif mode == 'GarbageClassification':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'TrafficDetection':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
        elif mode == 'ApriltagDiscern':
            client.test_set_mode(mode)
        elif mode == 'DMCodeDiscern':
            client.test_set_mode(mode)
        elif mode == 'QRCodeDiscern':
            client.test_set_mode(mode)
        elif mode == 'BarCodeDiscern':
            client.test_set_mode(mode)
        elif mode == 'CustomDetection':
            client.test_set_mode(mode)
            client.test_set_confidence_threshold(70)
            client.test_set_nms_threshold(45)
            client.test_set_custom_model(2)
        elif mode == 'AiLLM_Mode':
            # client.test_set_mode(mode)
            client.test_set_mcp_tools(mcp_tools)
            # client.test_set_llm_base_url('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')
            # client.test_set_llm_key('sk-db8141b1c35540738e77d0ae3c85fa21')
            # client.test_set_speech_url('wss://dashscope.aliyuncs.com/api-ws/v1/inference')
            # client.test_set_llm_model('qwen3.5-plus')
            # client.test_set_vlm_base_url('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')
            # client.test_set_vlm_model('qwen-vl-max-2025-08-13')
            # client.test_set_tts_voice('cosyvoice-v3-flash', 'longhuhu_v3')
            # client.test_set_asr_language('zh')
            # client.test_set_thinking_mode(False)
            # client.test_set_search_mode(False)
            # client.test_llm_chat('简单说下明天深圳天气')
            # client.test_set_start_silence(2000)
            # client.test_set_end_silence(1200)
            # client.test_set_system_prompt('你是一个助手。')
            # client.test_start_asr(True)
            # client.test_start_asr(False)
            # client.test_tts_speak('你好，我是小幻。')
            # client.test_llm_chat('讲个笑话')
            # client.test_vlm_chat('简单描述当前画面')

            # client.test_send_mcp_result(mcp_result)
        elif mode == 'MediaEnterCameraApp':
            auto_enable_run = False
            client.test_media_enter_camera_app()
        elif mode == 'MediaSetPhotoPrefix':
            auto_enable_run = False
            client.test_media_set_photo_prefix('picture_')
            # client.test_media_set_photo_prefix('')
        elif mode == 'MediaSetPhotoStart':
            auto_enable_run = False
            client.test_media_set_photo_start(0)
        elif mode == 'MediaCameraSnapshot':
            auto_enable_run = False
            client.test_media_set_photo_prefix('picture_')
            client.test_media_camera_snapshot()
        elif mode == 'MediaDeletePhoto':
            auto_enable_run = False
            client.test_media_delete_photo('picture_1')
            # client.test_media_delete_photo('picture_1.png')
        else:
            raise ValueError("不支持的测试模式: %s" % mode)

        if auto_enable_run:
            client.test_set_run_enabled(True)
        _run_example_loop(
            client,
            post_enable_actions,
            detect_trigger_action,
            detect_trigger_state,
            mcp_reply_state=mcp_reply_state,
            trace_state=trace_state,
        )
        return 0

    except Exception as e:
        if I2CClient.is_ide_interrupt(e):
            print("IDE interrupt, exiting...")
            return 0
        else:
            print("Error: %s" % e)
            return 1
    finally:
        if client:
            client.deinit()


if __name__ == "__main__":
    main()
