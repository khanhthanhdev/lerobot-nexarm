/*
 * K230 MCP pure demo for wonderlens_system protocol v2.4.
 *
 * Flow:
 * 1. Initialize I2C mailbox v2.
 * 2. Send CMD_SET_MCP_TOOLS (0x6C) with one demo MCP tool.
 * 3. Listen for CMD_RESULT_RETURN (0x6D).
 * 4. Print the tool call data and send "执行成功" back.
 */

#include <Arduino.h>
#include <Wire.h>
#include <string.h>

static constexpr uint8_t K230_ADDR = 0x5F;
static constexpr int I2C_SDA_PIN = 21;
static constexpr int I2C_SCL_PIN = 22;

static constexpr uint16_t MAILBOX_SIZE = 4096;
static constexpr uint16_t MAILBOX_HEADER_SIZE = 32;
static constexpr uint16_t HOST_SLOT_META_OFFSET = 16;
static constexpr uint16_t DEV_SLOT_META_OFFSET = 24;
static constexpr uint16_t MAX_FRAME_SIZE = 2048;

static constexpr uint8_t FRAME_H0 = 0xAA;
static constexpr uint8_t FRAME_H1 = 0x55;
static constexpr uint8_t CTRL_TYPE_CMD = 0x00;
static constexpr uint8_t CTRL_TYPE_RSP = 0x40;
static constexpr uint8_t CTRL_TYPE_RPT = 0x80;
static constexpr uint8_t CTRL_CONT = 0x20;
static constexpr uint8_t CTRL_SEQ_MASK = 0x1F;

static constexpr uint8_t CMD_SET_MCP_TOOLS = 0x6C;
static constexpr uint8_t CMD_RESULT_RETURN = 0x6D;
static constexpr uint8_t RPT_HEARTBEAT = 0x70;

static constexpr uint8_t TYPE_NULL = 0x00;
static constexpr uint8_t TYPE_BOOL_F = 0x01;
static constexpr uint8_t TYPE_BOOL_T = 0x02;
static constexpr uint8_t TYPE_UINT8 = 0x06;
static constexpr uint8_t TYPE_STRING = 0x09;
static constexpr uint8_t TYPE_ARRAY = 0x0A;
static constexpr uint8_t TYPE_DICT = 0x0B;

static const uint8_t MAILBOX_MAGIC[4] = {0x57, 0x4C, 0x4D, 0x32}; // WLM2

enum SlotState : uint8_t {
  SLOT_EMPTY = 0,
  SLOT_WRITING = 1,
  SLOT_READY = 2,
};

struct SlotMeta {
  uint8_t state;
  uint8_t reserved0;
  uint16_t generation;
  uint16_t frame_len;
  uint8_t frame_xor;
  uint8_t reserved1;
};

class DataPacker {
public:
  uint8_t buf[2048];
  uint16_t len = 0;

  void addString(const char* s) {
    uint16_t slen = strlen(s);
    buf[len++] = TYPE_STRING;
    buf[len++] = (uint8_t)(slen >> 8);
    buf[len++] = (uint8_t)(slen & 0xFF);
    memcpy(&buf[len], s, slen);
    len += slen;
  }

  void addBool(bool value) {
    buf[len++] = value ? TYPE_BOOL_T : TYPE_BOOL_F;
  }

  void addUint8(uint8_t value) {
    buf[len++] = TYPE_UINT8;
    buf[len++] = value;
  }

  void beginArray(uint16_t count) {
    buf[len++] = TYPE_ARRAY;
    buf[len++] = (uint8_t)(count >> 8);
    buf[len++] = (uint8_t)(count & 0xFF);
  }

  void beginDict(uint16_t count) {
    buf[len++] = TYPE_DICT;
    buf[len++] = (uint8_t)(count >> 8);
    buf[len++] = (uint8_t)(count & 0xFF);
  }
};

static uint16_t slot_size = 0;
static uint16_t host_slot_data_offset = 0;
static uint16_t dev_slot_data_offset = 0;
static uint16_t host_gen = 0;
static uint16_t dev_gen = 0;
static uint8_t tx_txn = 0;
static uint8_t tx_seq = 0;

static bool rx_frag_active = false;
static uint8_t rx_frag_type = 0;
static uint8_t rx_frag_func = 0;
static uint8_t rx_frag_txn = 0;
static uint8_t rx_frag_expected_seq = 0;
static uint16_t rx_frag_len = 0;
static uint8_t rx_frag_buf[MAX_FRAME_SIZE];

static void i2cWrite16(uint16_t memAddr, const uint8_t* data, uint16_t len) {
  const uint16_t chunk = 30;
  for (uint16_t off = 0; off < len;) {
    uint16_t n = min(chunk, (uint16_t)(len - off));
    Wire.beginTransmission(K230_ADDR);
    Wire.write((uint8_t)((memAddr + off) >> 8));
    Wire.write((uint8_t)((memAddr + off) & 0xFF));
    Wire.write(data + off, n);
    Wire.endTransmission();
    off += n;
  }
}

static void i2cRead16(uint16_t memAddr, uint8_t* data, uint16_t len) {
  const uint16_t chunk = 32;
  for (uint16_t off = 0; off < len;) {
    uint16_t n = min(chunk, (uint16_t)(len - off));
    Wire.beginTransmission(K230_ADDR);
    Wire.write((uint8_t)((memAddr + off) >> 8));
    Wire.write((uint8_t)((memAddr + off) & 0xFF));
    Wire.endTransmission(false);
    Wire.requestFrom((uint16_t)K230_ADDR, (uint8_t)n);
    for (uint16_t i = 0; i < n && Wire.available(); i++) {
      data[off + i] = Wire.read();
    }
    off += n;
  }
}

static SlotMeta readSlotMeta(uint16_t offset) {
  uint8_t raw[8];
  i2cRead16(offset, raw, sizeof(raw));
  SlotMeta meta;
  meta.state = raw[0];
  meta.reserved0 = raw[1];
  meta.generation = ((uint16_t)raw[2] << 8) | raw[3];
  meta.frame_len = ((uint16_t)raw[4] << 8) | raw[5];
  meta.frame_xor = raw[6];
  meta.reserved1 = raw[7];
  return meta;
}

static void writeSlotMeta(uint16_t offset, const SlotMeta& meta) {
  uint8_t raw[8] = {
    meta.state,
    meta.reserved0,
    (uint8_t)(meta.generation >> 8),
    (uint8_t)(meta.generation & 0xFF),
    (uint8_t)(meta.frame_len >> 8),
    (uint8_t)(meta.frame_len & 0xFF),
    meta.frame_xor,
    meta.reserved1,
  };
  i2cWrite16(offset, raw, sizeof(raw));
}

static bool initMailbox() {
  uint8_t header[8];
  i2cRead16(0, header, sizeof(header));
  if (memcmp(header, MAILBOX_MAGIC, sizeof(MAILBOX_MAGIC)) != 0) {
    Serial.println("[MCP] mailbox magic mismatch");
    return false;
  }

  slot_size = ((uint16_t)header[6] << 8) | header[7];
  if (slot_size <= 8 || slot_size > (MAILBOX_SIZE - MAILBOX_HEADER_SIZE) / 2) {
    Serial.printf("[MCP] bad slot_size=%d\n", slot_size);
    return false;
  }

  host_slot_data_offset = MAILBOX_HEADER_SIZE;
  dev_slot_data_offset = MAILBOX_HEADER_SIZE + slot_size;
  host_gen = readSlotMeta(HOST_SLOT_META_OFFSET).generation;
  dev_gen = readSlotMeta(DEV_SLOT_META_OFFSET).generation;
  Serial.printf("[MCP] mailbox ok slot_size=%d\n", slot_size);
  return true;
}

static bool writeHostSlot(const uint8_t* frame, uint16_t len) {
  if (slot_size == 0 || len > slot_size) {
    return false;
  }

  SlotMeta hm = readSlotMeta(HOST_SLOT_META_OFFSET);
  if (hm.state != SLOT_EMPTY) {
    return false;
  }

  host_gen = (host_gen + 1) & 0xFFFF;
  if (host_gen == 0) host_gen = 1;

  uint8_t xor_val = 0;
  for (uint16_t i = 0; i < len; i++) xor_val ^= frame[i];

  SlotMeta writing = {SLOT_WRITING, 0, host_gen, len, xor_val, 0};
  writeSlotMeta(HOST_SLOT_META_OFFSET, writing);
  i2cWrite16(host_slot_data_offset, frame, len);
  writing.state = SLOT_READY;
  writeSlotMeta(HOST_SLOT_META_OFFSET, writing);
  return true;
}

static int readDevSlot(uint8_t* buf, uint16_t bufSize) {
  SlotMeta dm = readSlotMeta(DEV_SLOT_META_OFFSET);
  if (dm.state != SLOT_READY) return 0;

  if (dm.generation == dev_gen) {
    SlotMeta ack = {SLOT_EMPTY, 0, dm.generation, 0, 0, 0};
    writeSlotMeta(DEV_SLOT_META_OFFSET, ack);
    return 0;
  }

  uint16_t frameLen = dm.frame_len;
  if (frameLen < 8 || frameLen > bufSize || frameLen > slot_size) {
    SlotMeta ack = {SLOT_EMPTY, 0, dm.generation, 0, 0, 0};
    writeSlotMeta(DEV_SLOT_META_OFFSET, ack);
    dev_gen = dm.generation;
    Serial.printf("[MCP] bad dev frame len=%d\n", frameLen);
    return 0;
  }

  i2cRead16(dev_slot_data_offset, buf, frameLen);
  SlotMeta ack = {SLOT_EMPTY, 0, dm.generation, 0, 0, 0};
  writeSlotMeta(DEV_SLOT_META_OFFSET, ack);
  dev_gen = dm.generation;

  uint8_t xor_val = 0;
  for (uint16_t i = 0; i < frameLen; i++) xor_val ^= buf[i];
  if (xor_val != dm.frame_xor) {
    Serial.printf("[MCP] dev slot xor fail calc=%02X got=%02X\n", xor_val, dm.frame_xor);
    return -1;
  }
  return frameLen;
}

static uint8_t nextTxn() {
  tx_txn++;
  if (tx_txn == 0) tx_txn = 1;
  return tx_txn;
}

static uint8_t nextSeq() {
  uint8_t seq = tx_seq & CTRL_SEQ_MASK;
  tx_seq = (tx_seq + 1) & CTRL_SEQ_MASK;
  return seq;
}

static uint16_t buildFrame(uint8_t* buf,
                           uint8_t func,
                           const uint8_t* payload,
                           uint16_t plen,
                           uint8_t txn,
                           uint8_t seq,
                           bool continuation) {
  buf[0] = FRAME_H0;
  buf[1] = FRAME_H1;
  buf[2] = (uint8_t)(plen >> 8);
  buf[3] = (uint8_t)(plen & 0xFF);
  buf[4] = CTRL_TYPE_CMD | (continuation ? CTRL_CONT : 0) | (seq & CTRL_SEQ_MASK);
  buf[5] = func;
  buf[6] = txn;
  if (payload && plen > 0) {
    memcpy(&buf[7], payload, plen);
  }

  uint8_t xor_val = 0;
  for (uint16_t i = 2; i < 7 + plen; i++) xor_val ^= buf[i];
  buf[7 + plen] = xor_val;
  return 8 + plen;
}

static bool sendCmd(uint8_t func, const uint8_t* payload = nullptr, uint16_t plen = 0) {
  if (slot_size <= 8 || plen > MAX_FRAME_SIZE - 8) {
    return false;
  }

  uint16_t maxPayload = min((uint16_t)(MAX_FRAME_SIZE - 8), (uint16_t)(slot_size - 8));
  uint16_t offset = 0;
  uint8_t txn = nextTxn();
  bool sentAny = false;

  do {
    uint16_t chunkLen = (plen == 0) ? 0 : min(maxPayload, (uint16_t)(plen - offset));
    bool continuation = (offset + chunkLen) < plen;
    bool slotReady = false;

    for (int i = 0; i < 40; i++) {
      if (readSlotMeta(HOST_SLOT_META_OFFSET).state == SLOT_EMPTY) {
        slotReady = true;
        break;
      }
      delay(25);
    }
    if (!slotReady) {
      Serial.printf("[MCP] host slot busy func=0x%02X off=%d\n", func, offset);
      return false;
    }

    uint8_t frame[MAX_FRAME_SIZE];
    uint16_t frameLen = buildFrame(frame,
                                   func,
                                   payload ? payload + offset : nullptr,
                                   chunkLen,
                                   txn,
                                   nextSeq(),
                                   continuation);
    if (!writeHostSlot(frame, frameLen)) {
      Serial.printf("[MCP] send failed func=0x%02X off=%d len=%d\n", func, offset, chunkLen);
      return false;
    }

    sentAny = true;
    offset += chunkLen;
  } while (offset < plen);

  return sentAny;
}

static void dumpBytes(const char* label, const uint8_t* data, uint16_t len, uint16_t limit = 192) {
  Serial.printf("[MCP] %s len=%d\n", label, len);
  uint16_t n = min(len, limit);
  for (uint16_t i = 0; i < n; i++) {
    if ((i % 16) == 0) Serial.printf("[MCP]   %04X: ", i);
    Serial.printf("%02X ", data[i]);
    if ((i % 16) == 15 || i + 1 == n) Serial.println();
  }
  if (len > limit) Serial.println("[MCP]   ...");
}

static void printDataPackStrings(const uint8_t* data, uint16_t len) {
  bool found = false;
  for (uint16_t i = 0; i + 3 <= len; i++) {
    if (data[i] != TYPE_STRING) continue;
    uint16_t slen = ((uint16_t)data[i + 1] << 8) | data[i + 2];
    if (slen == 0 || i + 3 + slen > len) continue;

    Serial.printf("[MCP] string @%d (%d): ", i, slen);
    for (uint16_t j = 0; j < slen; j++) {
      uint8_t c = data[i + 3 + j];
      if (c >= 0x20 && c <= 0x7E) Serial.write(c);
      else Serial.printf("\\x%02X", c);
    }
    Serial.println();
    found = true;
  }
  if (!found) {
    Serial.println("[MCP] no data_pack strings found");
  }
}

static bool containsBytes(const uint8_t* data, uint16_t len, const char* text) {
  uint16_t textLen = strlen(text);
  if (textLen == 0 || len < textLen) return false;
  for (uint16_t i = 0; i <= len - textLen; i++) {
    if (memcmp(&data[i], text, textLen) == 0) return true;
  }
  return false;
}

static void sendMcpSuccessResult(const char* toolName) {
  DataPacker p;
  p.addString("执行成功");

  bool ok = sendCmd(CMD_RESULT_RETURN, p.buf, p.len);
  Serial.printf("[MCP] result return sent ok=%d tool=%s payload=%d\n", ok, toolName, p.len);
}

static void handleMcpResultRsp(const uint8_t* payload, uint16_t plen) {
  if (plen < 4) {
    Serial.printf("[MCP] RSP 0x6D too short plen=%d\n", plen);
    return;
  }

  uint8_t err = payload[0];
  uint8_t errModule = payload[1];
  uint16_t errSubcode = ((uint16_t)payload[2] << 8) | payload[3];
  const uint8_t* extra = &payload[4];
  uint16_t extraLen = plen - 4;

  Serial.printf("[MCP] RSP 0x6D err=0x%02X module=0x%02X sub=0x%04X extra=%d\n",
                err, errModule, errSubcode, extraLen);
  if (err != 0 || extraLen == 0) return;

  dumpBytes("MCP tool raw extra", extra, extraLen);
  printDataPackStrings(extra, extraLen);

  const char* toolName = "unknown";
  if (containsBytes(extra, extraLen, "move_arm")) {
    toolName = "move_arm";
  } else if (containsBytes(extra, extraLen, "demo_action")) {
    toolName = "demo_action";
  }

  Serial.printf("[MCP] execute tool=%s\n", toolName);
  sendMcpSuccessResult(toolName);
}

static void resetRxFragment() {
  rx_frag_active = false;
  rx_frag_type = 0;
  rx_frag_func = 0;
  rx_frag_txn = 0;
  rx_frag_expected_seq = 0;
  rx_frag_len = 0;
}

static void parseFrame(const uint8_t* frame, uint16_t len) {
  if (len < 8 || frame[0] != FRAME_H0 || frame[1] != FRAME_H1) {
    dumpBytes("bad frame", frame, len, 64);
    return;
  }

  uint16_t plen = ((uint16_t)frame[2] << 8) | frame[3];
  if (8 + plen > len) {
    Serial.printf("[MCP] frame length mismatch plen=%d raw=%d\n", plen, len);
    return;
  }

  uint8_t xor_val = 0;
  for (uint16_t i = 2; i < 7 + plen; i++) xor_val ^= frame[i];
  if (xor_val != frame[7 + plen]) {
    Serial.printf("[MCP] frame xor fail calc=%02X got=%02X\n", xor_val, frame[7 + plen]);
    return;
  }

  uint8_t ctrl = frame[4];
  uint8_t ftype = ctrl & 0xC0;
  bool continuation = (ctrl & CTRL_CONT) != 0;
  uint8_t seq = ctrl & CTRL_SEQ_MASK;
  uint8_t func = frame[5];
  uint8_t txn = frame[6];
  const uint8_t* payload = &frame[7];

  if (!(ftype == CTRL_TYPE_RPT && func == RPT_HEARTBEAT)) {
    Serial.printf("[MCP] RX type=0x%02X func=0x%02X txn=%d seq=%d cont=%d plen=%d\n",
                  ftype, func, txn, seq, continuation, plen);
  }

  if (continuation || rx_frag_active) {
    if (!rx_frag_active) {
      rx_frag_active = true;
      rx_frag_type = ftype;
      rx_frag_func = func;
      rx_frag_txn = txn;
      rx_frag_expected_seq = seq;
      rx_frag_len = 0;
    } else if (ftype != rx_frag_type ||
               func != rx_frag_func ||
               txn != rx_frag_txn ||
               seq != rx_frag_expected_seq) {
      Serial.println("[MCP] RX fragment mismatch");
      resetRxFragment();
      return;
    }

    if ((uint32_t)rx_frag_len + plen > sizeof(rx_frag_buf)) {
      Serial.println("[MCP] RX fragment overflow");
      resetRxFragment();
      return;
    }

    memcpy(&rx_frag_buf[rx_frag_len], payload, plen);
    rx_frag_len += plen;
    rx_frag_expected_seq = (seq + 1) & CTRL_SEQ_MASK;

    if (continuation) return;

    payload = rx_frag_buf;
    plen = rx_frag_len;
    ftype = rx_frag_type;
    func = rx_frag_func;
    resetRxFragment();
  }

  if (ftype == CTRL_TYPE_RSP) {
    uint8_t err = plen > 0 ? payload[0] : 0xFF;
    Serial.printf("[MCP] RSP func=0x%02X err=0x%02X plen=%d\n", func, err, plen);
    if (func == CMD_RESULT_RETURN) {
      handleMcpResultRsp(payload, plen);
    }
  } else if (ftype == CTRL_TYPE_RPT && func == RPT_HEARTBEAT && plen >= 2) {
    static uint8_t lastMode = 0xFF;
    static uint8_t lastStatus = 0xFF;
    if (payload[0] != lastMode || payload[1] != lastStatus) {
      lastMode = payload[0];
      lastStatus = payload[1];
      Serial.printf("[MCP] HB mode=%d status=0x%02X\n", lastMode, lastStatus);
    }
  }
}

static void setupMcpTools() {
  DataPacker p;
  p.beginArray(1);
    p.beginDict(3);
      p.addString("type");
      p.addString("function");
      p.addString("function");
      p.beginDict(3);
        p.addString("name");
        p.addString("move_arm");
        p.addString("description");
        p.addString("当用户要求机械臂动作时必须调用此工具，包括点头、摇头、抬头、低头、向左看、向右看、张开爪子、闭合爪子。");
        p.addString("parameters");
        p.beginDict(3);
          p.addString("type");
          p.addString("object");
          p.addString("properties");
          p.beginDict(1);
            p.addString("name");
            p.beginDict(2);
              p.addString("type");
              p.addString("string");
              p.addString("description");
              p.addString("动作名称，可选 nod、shake、look_up、look_down、look_left、look_right、open_claw、close_claw。");
          p.addString("required");
          p.beginArray(1);
            p.addString("name");
      p.addString("block");
      p.addUint8(5);

  bool ok = sendCmd(CMD_SET_MCP_TOOLS, p.buf, p.len);
  Serial.printf("[MCP] CMD_SET_MCP_TOOLS sent ok=%d payload=%d\n", ok, p.len);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("==== K230 Pure MCP Demo ====");

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000);
  Wire.setBufferSize(256);

  if (!initMailbox()) {
    Serial.println("[MCP] mailbox init failed");
    return;
  }

  setupMcpTools();
}

void loop() {
  uint8_t frame[MAX_FRAME_SIZE];
  int got = readDevSlot(frame, sizeof(frame));
  if (got > 0) {
    parseFrame(frame, (uint16_t)got);
  }
  delay(20);
}
