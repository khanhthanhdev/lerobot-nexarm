# wondermk 外部通信协议规范（I2C / UART）

## 1. 文档定位

### 1.1 适用范围

本文档定义 `wondermk` 对外 I2C / UART 通信协议（协议主版本 `2`、次版本 `6`）的正式规范，覆盖：

- 公共帧格式与分包重组
- UART 传输语义
- I2C mailbox v2 语义
- 命令（CMD）与响应（RSP）
- 主动上报（RPT）
- 检测结果 payload 结构

### 1.2 目标读者

- 主机端协议栈开发人员（MCU / Linux / Windows / Android）
- 联调、测试与集成工程师
- 二次开发人员

### 1.3 非适用范围

- 不包含其它协议主版本的迁移说明
- 不包含 UI 交互设计说明
- 不包含具体 AI 模型内部算法说明

## 2. 规范性依据与实现基线

本文档基于仓库中当前实际生效实现整理，主要核对以下文件：

- 主机参考实现
  - `protocol_common.py`
  - `uart_client.py`
  - `i2c_client.py`

## 3. 一致性关键字与术语

### 3.1 一致性关键字

本文档使用以下关键字表达约束强度：

- `MUST`：必须满足
- `MUST NOT`：禁止
- `SHOULD`：推荐满足（有充分理由可偏离）
- `MAY`：可选

### 3.2 术语

- `CMD`：主机发送给从机的命令帧
- `RSP`：从机针对命令返回的响应帧
- `RPT`：从机主动上报帧（心跳、错误、检测结果）
- `Txn`：事务号，主机命令使用 `1..255`，主动上报固定 `0`
- `extra`：`RSP` 负载中错误明细头（`err_code+err_module+err_subcode`）之后的附加数据
- `simple_result`：检测结果简化开关，由 `CMD_SET_SIMPLE_RESULT(0x13)` 控制

## 4. 协议架构总览

### 4.1 分层模型

I2C 与 UART 共用同一业务协议，差异仅在传输层：

- 帧层：统一 `AA55 + Len + Ctrl + Func + Txn + Payload + XOR`
- 业务层：统一命令码、响应码、上报码与 payload 编码
- 传输层：
  - UART：连续字节流
  - I2C：mailbox v2 共享窗口

### 4.2 当前默认参数

- 默认外部协议：I2C
- 默认 I2C 从机地址：`0x5F`
- I2C mailbox 总窗口：`4096` 字节
- 当前典型单帧最大长度：`256` 字节
- 协议理论最大帧长度：`4096` 字节

说明：主机端实际可发帧长 `MUST` 以 `CMD_GET_PROTOCOL_INFO(0x06)` 返回的 `max_frame_len` 为准。

## 5. 公共帧格式

### 5.1 字节布局

| 字段 | 长度 | 偏移 | 说明 |
| --- | ---: | ---: | --- |
| Header | 2 | 0 | 固定 `0xAA 0x55` |
| Len | 2 | 2 | `Payload` 长度，`uint16`，大端 |
| Ctrl | 1 | 4 | 帧类型 + 续包标志 + 分包序号 |
| Func | 1 | 5 | 功能码 |
| Txn | 1 | 6 | 事务号 |
| Payload | `Len` | 7 | 负载 |
| XOR | 1 | `7 + Len` | 校验 |

公式：

```text
Frame = AA 55 + Len(2) + Ctrl(1) + Func(1) + Txn(1) + Payload + XOR(1)
```

### 5.2 长度边界

- 最小帧长：`8`
- 理论最大帧长：`4096`
- 理论最大 payload：`4088`

当前正式固件常见单帧上限为 `256`，对应 payload 上限通常为 `248`。

### 5.3 Ctrl 位定义

| 位 | 含义 |
| --- | --- |
| bit7..6 | 帧类型 |
| bit5 | `1` 表示后续仍有分片 |
| bit4..0 | 分片序号 `0..31` |

掩码：

- `TYPE_MASK = 0xC0`
- `CONT_MASK = 0x20`
- `SEQ_MASK = 0x1F`

### 5.4 帧类型

| 值 | 名称 | 方向 | 说明 |
| --- | --- | --- | --- |
| `0x00` | CMD | 主机 -> 从机 | 命令 |
| `0x40` | RSP | 从机 -> 主机 | 命令响应 |
| `0x80` | RPT | 从机 -> 主机 | 主动上报 |
| `0xC0` | ACK | 预留 | 当前未启用 |

### 5.5 Txn 事务号规则

- 主机命令 `MUST` 使用 `Txn=1..255`
- 主机命令 `MUST NOT` 使用 `Txn=0`
- 从机主动上报固定 `Txn=0`
- 从机响应 `MUST` 回显请求 `Txn`
- 同一未完成事务 `MUST NOT` 复用 Txn

### 5.6 XOR 校验

XOR 计算范围不含 Header 和 XOR 本身：

```text
Len_hi ^ Len_lo ^ Ctrl ^ Func ^ Txn ^ Payload[0] ^ Payload[1] ...
```

## 6. 分包与重组

### 6.1 发送端分包规则

当 payload 超过链路 `max_frame_len - 8` 时：

- 发送方按序切分为多帧
- 非最后分片：`Ctrl.bit5=1`
- 最后分片：`Ctrl.bit5=0`
- 分片序号使用 `Ctrl.bit4..0`，按 `0..31` 回绕

### 6.2 接收端重组匹配项

重组阶段 `MUST` 校验以下字段一致：

- `frame_type`
- `func_code`
- `txn_id`
- `sequence` 连续性

### 6.3 失败与错误码

重组错误常见返回：

- `ERR_SEQ_MISMATCH (0x0B)`
- `ERR_REASSEMBLE_FAIL (0x0C)`

## 7. UART 传输层规范

UART 传输不引入额外包头，直接传输第 5 章定义的完整帧。

主机侧 `MUST` 实现：

- 帧头同步
- 长度提取
- XOR 校验
- 分包重组

主机侧 `MUST NOT` 假设：

- 一次 `read()` 必定读满一帧
- 帧间存在固定空隙

## 8. I2C mailbox v2 规范

### 8.1 总体结构

I2C 使用固定 `4096` 字节共享窗口，结构如下：

| 区域 | 偏移 | 长度 | 说明 |
| --- | ---: | ---: | --- |
| Mailbox Header | `0` | `32` | 固定头 |
| Host Slot Meta | `16` | `8` | 主机 -> 从机元数据 |
| Device Slot Meta | `24` | `8` | 从机 -> 主机元数据 |
| Slot Data Base | `32` | `slot_size * 2` | 两个数据槽 |

其中：

- Host Slot Data：偏移 `32`
- Device Slot Data：偏移 `32 + slot_size`

### 8.2 传输前提

mailbox v2 生效前提：

- 主机按 `16-bit memaddr` 访问偏移 `0..4095`
- 从机后端按两字节地址指针解释偏移
- 共享窗口完整暴露 `4096` 字节

仅能 `i2c scan` 到地址并不等价于 mailbox v2 可用。

### 8.3 Mailbox Header

| 偏移 | 长度 | 含义 |
| --- | ---: | --- |
| `0..3` | 4 | 固定魔数 `57 4C 4D 32`（`"WLM2"`） |
| `4` | 1 | mailbox 主版本，当前 `2` |
| `5` | 1 | mailbox 次版本，当前 `0` |
| `6..7` | 2 | `slot_size`（大端） |
| 其它 | - | 保留 |

### 8.4 Slot Meta 结构

每个 slot meta 固定 8 字节：

| 偏移 | 长度 | 含义 |
| --- | ---: | --- |
| `0` | 1 | `state` |
| `1` | 1 | 保留 |
| `2..3` | 2 | `generation`（大端） |
| `4..5` | 2 | `frame_len`（大端） |
| `6` | 1 | `frame_xor` |
| `7` | 1 | 保留 |

### 8.5 Slot 状态值

| 值 | 名称 | 含义 |
| --- | --- | --- |
| `0` | EMPTY | 空槽，可写 |
| `1` | WRITING | 正在写入 |
| `2` | READY | 已写完，待读取 |

### 8.6 主机写 Host Slot 流程

1. 等待 `Host Slot state == EMPTY`
2. `generation` 加 1
3. 写 Host Slot Meta，置 `state=WRITING`
4. 写完整协议帧到 Host Slot Data
5. 写 Host Slot Meta，置 `state=READY`，并填 `frame_len/frame_xor`
6. 等待从机消费后回到 `EMPTY`

### 8.7 主机读 Device Slot 流程

1. 轮询 Device Slot Meta，等待 `state=READY`
2. 若 `generation` 未变化，视为无新帧
3. 校验 `frame_len`
4. 读取 Device Slot Data
5. 校验 `frame_xor`
6. 写回 `state=EMPTY`，记录本次 `generation`

### 8.8 关键约束

主机侧：

- `MUST NOT` 观察前 2 字节 `00 00` 判空
- `MUST NOT` 直接向偏移 `0` 写协议帧
- `MUST` 通过 slot state 管理收发确认
- `MUST` 区分 Host Slot 与 Device Slot

## 9. 数据编码规范

### 9.1 CompactCodec

多数命令使用 CompactCodec：

| 类型 | 编码 |
| --- | --- |
| `string` | `[len:u8][utf8 bytes...]` |
| `uint8` | 1 字节 |
| `uint16` | 大端 2 字节 |
| `uint32` | 大端 4 字节 |
| `bbox` | `[center_x:u16][center_y:u16][w:u16][h:u16]` |

补充：

- `string` 协议长度上限为 `255` 字节
- 业务侧名称缓冲区多为 `32` 字节，建议名称不超过 `31` 字节
- 名称会做安全校验（禁止控制字符、`/ \\ | ; , = :` 等）

### 9.1.1 颜色选择状态同步

颜色相关命令成功后，从机内部会同步 UI 选中态：

- `CMD_COLOR_SET_TARGET`、颜色学习选择阈值、颜色学习重命名/删除会同步当前目标颜色名称
- `CMD_MULTI_COLOR_SET_LIST` 会同步多颜色名称列表
- 自定义颜色名称优先于内置颜色 ID；内置颜色 ID 仅作为无名称时的回退
- 同步仅影响设备 UI 与运行状态，不改变 I2C/UART 对外命令 payload 编码
- 当前设备内部 big-core 到 little-core 的 mode sync 会携带 `color_target_name`、`color_multi_count`、`color_multi_names`

### 9.2 颜色阈值编码

`CMD_COLOR_SET_THRESH` 单项阈值为 6 字节：

```text
[L_min:u8][L_max:u8][A_min:i8][A_max:i8][B_min:i8][B_max:i8]
```

### 9.3 巡线 ROI 编码

`CMD_LINE_SET_ROI` 固定 15 字节：

```text
3 * [x:u8][y:u8][w:u8][h:u8][weight:u8]
```

约束：

- `x/y/w/h/weight` 都是百分比值，范围 `0..100`，不是像素坐标。
- 每个 ROI 都要求 `w>0`、`h>0`，且 `x + w <= 100`、`y + h <= 100`。

### 9.4 `data_pack` 编码

`CMD_SET_MCP_TOOLS (0x6C)` 与 `CMD_RESULT_RETURN (0x6D)` 不使用 CompactCodec，使用 `data_pack` 类型化编码。

#### 9.4.1 类型标签

| Tag | 含义 | 后续数据 |
| --- | --- | --- |
| `0x00` | `null` | 无 |
| `0x01` | `false` | 无 |
| `0x02` | `true` | 无 |
| `0x03` | `int8` | 1 字节（有符号） |
| `0x04` | `int16` | 2 字节（大端有符号） |
| `0x05` | `int32` | 4 字节（大端有符号） |
| `0x06` | `uint8` | 1 字节 |
| `0x07` | `uint16` | 2 字节（大端） |
| `0x08` | `uint32` | 4 字节（大端） |
| `0x09` | `string` | `[len:u16][utf8...]` |
| `0x0A` | `array` | `[count:u16][item1][item2]...` |
| `0x0B` | `dict` | `[count:u16][key1][value1]...` |

#### 9.4.2 使用建议

- 主机下发 `CMD_SET_MCP_TOOLS(0x6C)` 时，MCP tools 建议使用 `array<dict>`
- 主机向设备回填工具执行结果时，`CMD_RESULT_RETURN(0x6D)` payload 建议使用 `dict`
- 从机向主机转发 MCP 工具调用时，会发送 `RSP func=0x6D`，`err_code=0`，`extra` 为 `data_pack(dict)`，推荐格式为 `{tool_name: args}`
- `RSP func=0x6D` 可能来自主机发起的 ASR / LLM / VLM 异步事务，也可能来自设备端唤醒后自主触发的 MCP 工具调用；后一种情况下 `txn` 由从机分配为非零值，不对应主机当前等待中的命令
- 主机收到 `RSP func=0x6D` 且 `err_code=0` 时，`extra` 应按 `data_pack` 解析；即使 `txn` 不匹配当前挂起命令，也应作为可处理的异步结果/工具调用分发给上层

### 9.5 媒体命名与删除规则

- 照片目录：`/sharefs/media/photos`
- 前缀配置：`/sharefs/hiwonder/media_photo_prefix.txt`

命名规则：

- 前缀文件不存在：默认前缀 `image_`（示例：`image_1.png`）
- 前缀文件存在且为空：时间戳命名
- 前缀文件存在且非空：`prefix + index + .png`
- 索引分配策略：扫描目录并补最小缺失正整数（非简单 `max+1`）

删除规则：

- `CMD_MEDIA_DELETE_PHOTO` payload 为单个 `string_u8`
- 允许传基础名或完整文件名（可自动补扩展名）
- 名称安全校验与普通业务名称一致

## 10. 响应与异步模型

### 10.1 `RSP` 响应帧

响应统一格式：

- `frame_type = RSP`
- `func_code = 原命令功能码`
- `txn = 原命令 txn`
- `payload = [err_code:u8][err_module:u8][err_subcode:u16_be][extra...]`

字段说明：

| 字段 | 长度 | 说明 |
| --- | ---: | --- |
| `err_code` | 1 | 主错误码（见第 16 章） |
| `err_module` | 1 | 错误模块（见第 16 章） |
| `err_subcode` | 2 | 错误子码，大端 |
| `extra` | N | 命令附加数据 |

常见约定：

- `err_code=0x00` 表示命令成功
- 成功时默认 `err_module=0x00`、`err_subcode=0x0000`
- 多数同步命令成功时 `extra=0x01`；`CMD_COLOR_GET_THRESH(0x42)` 成功时直接返回 6 字节阈值，不附加 `0x01`

### 10.2 `RPT` 主动上报帧

主动上报统一：

- `frame_type = RPT`
- `txn = 0`
- `func_code` 按上报类型区分

当前上报范围：

- `0x70` 心跳
- `0x71` 错误
- `0x72..0x7B` 检测结果

在 I2C mailbox v2 下，检测与心跳属于“最新状态流”，从机会合并被新状态替代的上报，主机不应依赖“每一帧中间态都必达”。

进一步约束：

- 心跳属于空闲保活帧，不与连续业务结果保持同频
- 检测结果与心跳可能分别被后续状态覆盖，主机只应消费“当前最新状态”
- 主机 `MUST NOT` 以“是否持续收到心跳”作为链路存活的唯一判据
- 主机 `SHOULD` 以“是否持续收到任意有效协议帧（RSP/RPT）”作为链路活跃判据

### 10.3 `CMD_REQUEST_STATUS (0x04)` 特殊行为

标准行为：

1. 先返回同步 `RSP func=0x04`
2. 再主动发送最新状态心跳 `RPT func=0x70`

说明：协议建议 payload 为空；当前固件对多余 payload 长度不做强校验。

### 10.4 `CMD_CLEAR_MEMORY (0x05)` 语义

该命令清理协议结果状态，不重置业务模式。

会清理：

- 命令结果队列
- 异步结果状态
- 检测结果缓存

不会重置：

- 当前 `mode`
- 当前 `run` 状态
- `simple_result` 开关

### 10.5 异步命令两阶段返回

常见两阶段模式：

1. 原命令先收到同步 `RSP`（已受理）
2. 稍后通过最终 `RSP` 回传执行结果

当前常见最终功能码：

| 最终 `func` | 含义 |
| --- | --- |
| `0x6D` | `CMD_RESULT_RETURN`（ASR / LLM / VLM / MCP 结果） |
| `0x6E` | `CMD_EMPTY_RETURN`（Wi-Fi、TTS 完成、部分学习完成） |

补充约束：

- 设备端自主唤醒进入 AI 对话后，如果大模型调用了主机通过 `CMD_SET_MCP_TOOLS(0x6C)` 注册的外部 MCP 工具，从机会主动排队一个 `RSP func=0x6D` 给主机
- 该主动 MCP 回调仍使用 `RSP` 而不是 `RPT`，用于复用现有 `CMD_RESULT_RETURN` 解析；主机不得要求其 `txn` 必须等于某条未完成主机命令

## 11. 心跳与错误上报

### 11.1 `RPT_HEARTBEAT (0x70)`

Payload 固定 2 字节：

```text
[mode_index:u8][status_flags:u8]
```

状态位定义：

| Bit | 值 | 含义 |
| --- | --- | --- |
| bit0 | `0x01` | RUN |
| bit1 | `0x02` | RESULT |
| bit2 | `0x04` | READY |
| bit3 | `0x08` | BUSY |
| bit4 | `0x10` | ERR |

发送策略：

- `CMD_REQUEST_STATUS(0x04)` 成功后，设备 `MUST` 立即补发一帧最新状态心跳
- 发生模式切换、运行状态切换或其它 `state_sync_pending` 场景时，设备 `MUST` 立即补发一帧最新状态心跳
- 周期性心跳仅作为空闲保活，设备 `MUST NOT` 在同一轮已成功发送业务数据时再追加心跳
- 当前正式固件空闲心跳周期默认约为 `500ms`

主机实现建议：

- 若持续收到检测结果或命令响应，主机 `MUST` 视为链路存活，即使此时没有独立心跳
- 心跳更适合用于读取“当前 mode/status 快照”，不应用于估算业务结果帧率

### 11.2 `RPT_ERROR (0x71)`

Payload：

```text
[error_code:u8][error_module:u8][error_subcode:u16_be][module:string_u8][message:string_u8]
```

参考解析：

```json
{
  "code": 8,
  "module_id": 6,
  "subcode": 1026,
  "module": "ocr",
  "message": "publish failed"
}
```

## 12. 检测结果上报规范

### 12.1 解析总则

- 主机 `MUST` 按 `func_code` 解析结果，不按 app 名硬编码
- `simple_result`：`0=完整模式`，`1=简化模式`
- 本章覆盖 `RPT 0x72..0x7B`

### 12.2 检测结果功能码总览

| Func | 名称 | 类型 |
| --- | --- | --- |
| `0x72` | `RPT_DETECT_BBOX` | 矩形框 |
| `0x73` | `RPT_DETECT_STR` | 字符串 |
| `0x74` | `RPT_DETECT_OCR` | 四点文本 |
| `0x75` | `RPT_DETECT_COLOR` | 颜色分组/旋转框 |
| `0x76` | `RPT_DETECT_LINE` | 巡线 |
| `0x77` | `RPT_DETECT_KEYPOINT` | 人体关键点 |
| `0x78` | `RPT_DETECT_HAND_KP` | 手掌关键点 |
| `0x79` | `RPT_DETECT_CENTER` | 中心点 |
| `0x7A` | `RPT_DETECT_FACE_KP` | 人脸五点 |
| `0x7B` | `RPT_DETECT_QUAD` | 四边形 |

### 12.3 几何类结果（BBOX / CENTER / QUAD）

几何坐标统一约定：

- 除特别说明外，只要结果中同时带有 `center_x/center_y/w/h`，其中 `center_x/center_y` 都表示目标中心点坐标，`w/h` 保持对应目标框的宽和高。
- 历史代码或说明中若仍出现 `x/y`，在几何结果语境下也应按“中心点坐标”理解，不再表示左上角。

#### 12.3.1 `RPT_DETECT_BBOX (0x72)`

Payload：

```text
[count:u8] +
count * (
  [center_x:u16][center_y:u16][w:u16][h:u16][extra_count:u8] +
  extra_count * extra_item
)
```

`BBOX extra_item` 类型：

| `type` | 编码 | 语义 |
| ---: | --- | --- |
| 1 | `[1][string_u8]` | 字符串 |
| 2 | `[2][value:s16]` | 定点浮点（实际值=`value/100`） |
| 3 | `[3][value:s16]` | 整数 |
| 4 | `[4][list_len:u8][list_len*s16]` | 整数数组 |

`center_x/center_y/w/h` 说明：

- 默认语义：`center_x/center_y` 为目标中心点，`w/h` 为目标框宽高。
- `EyeGaze` 是特例：`center_x/center_y` 直接等于注视起点 `cx/cy`，但 `w/h` 仍表示对应人脸框宽高。
- 除特别说明外，文中出现的 `score` / `score_pct` 都表示百分比整数，范围 `0..100`，编码为 `s16`。
- `score` 只会出现在本节明确列出的完整模式 `extra`、关键点结果或 `FalldownDetection` 简化模式里；颜色/巡线结果不返回 `score`。

完整模式常见语义：

| 模式/场景 | `extra_count` | `extra` 顺序与含义 |
| --- | ---: | --- |
| FaceDetection | 0 | 无 |
| FaceLandmark / FaceMesh | 10 | 5 点关键点坐标（10 个 `s16`） |
| FaceRecognition / FaceLiveness | 0 或 2 | 未识别/未命中时无；识别到已知目标时为 `[name:string][score:int]` |
| FacePose | 3~4 | `[id:string?][pitch:int][yaw:int][roll:int]` |
| FaceParse | 0 | 无 |
| EyeGaze | 4~5 | `[id:string?][cx:int][cy:int][tx:int][ty:int]` |
| PersonDetection | 0 | 无 |
| HandGesture / HandRecognition | 1 | `[gesture:string]`（固定英文枚举：`ok/fist/five/gun/other/love/one/six/three/thumbUp/yeah/unknown`） |
| FalldownDetection | 2 | `[fall_flag:int][score:int]`（`fall_flag`：`1=Fall`，`0=NoFall/其他`） |
| ObjectTrack | 0 | 无 |
| SelfLearning | 2 | `[name:string][score:int]` |
| TrafficDetection | 2 | `[label:string][score:int]`（见下方 `TrafficDetection` 标签集合） |
| QRCodeDiscern / BarCodeDiscern | 1 | `[payload:string]` |
| HandKeyPointDetection + `CMD_HAND_DETECT_ONLY=1` | 0 | 无 |
| LicencePlateDetection | 0 | 无文本，仅矩形框 |
| CustomDetection + `yolo_task=detect` | 2 | `[label:string][score:int]` |

`HandGesture / HandRecognition` 标签集合（用于 `gesture` 字段）：

| 序号 | 协议标签（固定英文） | 中文语义 |
| ---: | --- | --- |
| 0 | ok | OK 手势 |
| 1 | fist | 握拳 |
| 2 | five | 五指张开 |
| 3 | gun | 手枪手势 |
| 4 | other | 其他手势（手势识别） |
| 5 | love | 爱心手势 |
| 6 | one | 数字一 |
| 7 | six | 数字六 |
| 8 | three | 数字三 |
| 9 | thumbUp | 点赞 |
| 10 | yeah | V 手势 |
| 11 | unknown | 未识别/不确定 |

语言行为：

- 协议字段 `gesture` 固定使用上表英文 token
- 中文仅用于设备端本地显示映射，不改变协议返回值

`FalldownDetection` 状态集合（用于 `fall_flag` 字段）：

| 字段 | 取值 | 含义 |
| --- | ---: | --- |
| `fall_flag` | 1 | 跌倒（`Fall`） |
| `fall_flag` | 0 | 非跌倒（`NoFall`/其他） |

`FalldownDetection` 可在同一帧上报多个目标，每个目标各自携带 `[fall_flag][score]`。

`TrafficDetection` 标签集合（用于 `label` 字段）：

| 序号 | 协议标签（固定英文） | 中文语义 |
| ---: | --- | --- |
| 0 | red_barrier | 红色路障 |
| 1 | go_straight | 直行 |
| 2 | turn_left | 左转 |
| 3 | turn_right | 右转 |
| 4 | roundabout | 环岛 |
| 5 | parking_area | 停车区 |
| 6 | stop_sign | 停止标志 |
| 7 | traffic_light_red | 红灯 |
| 8 | traffic_light_yellow | 黄灯 |
| 9 | traffic_light_green | 绿灯 |
| 10 | pedestrian_crossing | 人行横道 |

语言行为：

- 协议字段 `label` 固定使用上表英文键
- 中文仅用于设备端本地显示映射，不改变协议返回值

#### 12.3.2 `RPT_DETECT_CENTER (0x79)`

Payload：

```text
[count:u8] +
count * (
  [center_x:u16][center_y:u16][extra_count:u8] +
  extra_count * extra_item
)
```

`CENTER extra_item` 类型：

| `type` | 编码 | 语义 |
| ---: | --- | --- |
| 1 | `[1][string_u8]` | 字符串 |
| 2 | `[2][value:s16]` | 整数 |
| 3 | `[3][value:s16]` | 定点浮点（实际值=`value/100`） |

简化模式（以及 `CustomDetection` 配置为 `yolo_task=cls`）常见语义：

| 模式/场景 | `center_x/center_y` 语义 | `extra` 顺序与含义 |
| --- | --- | --- |
| FaceDetection / FaceLandmark / FaceMesh | 中心点 | 无 |
| FaceRecognition / FaceLiveness | 中心点 | 无或 `[name]` |
| FacePose | 中心点 | `[id?][roll][pitch][yaw]` |
| FaceParse | 中心点 | 无 |
| EyeGaze | `x=cx, y=cy` | `[id?][tx][ty]` |
| PersonDetection / ObjectTrack | 中心点 | 无 |
| HandGesture / HandRecognition | 中心点 | `[gesture]`（枚举同 `12.3.1`） |
| FalldownDetection | 中心点 | `[fall_flag][score]`（`fall_flag`：`1=Fall`，`0=NoFall/其他`） |
| SingleColor / MultiColor / ColorTracking | 色块中心点 | `[color_name]` |
| LineDetection | `center_x=center_pos, center_y=0` | `[angle]` |
| OCRDetection / LicencePlateDetection | 目标中心点 | 无文本 |
| OCRRecognition / LicencePlateRecognition | 目标中心点 | `[text]`（无文本时可仅发送中心点，或按具体模式丢弃本帧） |
| GarbageClassification | 目标中心点 | `[label]`（枚举同 `12.3.3`） |
| TrafficDetection | 目标中心点 | `[label]`（枚举同 `12.3.1`） |
| ApriltagDiscern | 目标中心点 | `[family][tag_id?]`（仅可解析时含 `tag_id`） |
| DMCodeDiscern | 目标中心点 | `[payload]` |
| CustomDetection（简化或配置为 `yolo_task=cls`） | 目标中心点 | `[label]` |

#### 12.3.3 `RPT_DETECT_QUAD (0x7B)`

Payload：

```text
[count:u8] +
count * (
  [x0:s16][y0:s16][x1:s16][y1:s16][x2:s16][y2:s16][x3:s16][y3:s16] +
  [extra_count:u8] +
  extra_count * extra_item
)
```

其中 `extra_item` 编码与 `RPT_DETECT_BBOX` 一致。

完整模式常见语义：

| 模式/场景 | `extra_count` | `extra` 顺序与含义 |
| --- | ---: | --- |
| GarbageClassification | 2 | `[label:string][score:int]`（见下方 `GarbageClassification` 标签集合） |
| ApriltagDiscern | 1~2 | `[family:string][tag_id:int?]` |
| DMCodeDiscern | 1 | `[payload:string]` |
| OCRDetection | 0 | 无文本，仅四点框 |
| CustomDetection + `yolo_task=obb` | 2 | `[label:string][score:int]` |

`GarbageClassification` 标签集合（用于 `label` 字段）：

| 序号 | 协议标签（固定英文） | 中文语义 |
| ---: | --- | --- |
| 0 | BananaPeel | 香蕉皮 |
| 1 | BrokenBones | 碎骨头 |
| 2 | CigaretteEnd | 烟头 |
| 3 | DisposableChopsticks | 一次性筷子 |
| 4 | Ketchup | 番茄酱 |
| 5 | Marker | 记号笔 |
| 6 | OralLiquidBottle | 口服液瓶 |
| 7 | Plate | 盘子 |
| 8 | PlasticBottle | 塑料瓶 |
| 9 | StorageBattery | 蓄电池 |
| 10 | Toothbrush | 牙刷 |
| 11 | Umbrella | 雨伞 |

语言行为：

- 协议字段 `label` 固定使用上表英文键
- 中文仅用于设备端本地显示映射，不改变协议返回值

### 12.4 文本类结果（STR / OCR）

#### 12.4.1 `RPT_DETECT_STR (0x73)`

支持两种载荷：

- 单字符串：`[string_u8]`
- 字符串列表：`[count:u8][string_u8]...`

使用场景：

| 模式/场景 | 载荷形式 |
| --- | --- |
| DynamicGesture（完整） | 单字符串 |
| DynamicGesture（简化） | 字符串列表（`count=1`） |
| SelfLearning（简化） | 字符串列表（候选名称） |
| QRCodeDiscern / BarCodeDiscern（简化） | 字符串列表（码文本） |

#### 12.4.2 `RPT_DETECT_OCR (0x74)`

Payload：

```text
[count:u8][with_text:u8] +
count * (
  [x0:s16][y0:s16][x1:s16][y1:s16][x2:s16][y2:s16][x3:s16][y3:s16] +
  if with_text != 0: [text:string_u8]
)
```

使用场景：

| 模式/场景 | 完整模式 | 简化模式 |
| --- | --- | --- |
| OCRRecognition | `0x74` | `0x79`（`extra=text`） |
| LicencePlateRecognition | `0x74` | `0x79`（`extra=plate_text`） |

说明：

- `OCRDetection(mode=25)` 是 OCR 检测-only 分支，完整模式使用 `0x7B`，简化模式使用 `0x79`，均不携带文本。
- `OCRRecognition(mode=26)` 完整模式使用 `0x74`；仅空白字符组成的 OCR 文本会在设备端丢弃，若本帧无任何有效文本，则不发送检测 `RPT`。
- `LicencePlateRecognition(mode=28)` 完整模式在无结果时回退 `0x72` 空框；简化模式有文本时携带 `text`，无文本时仅发送中心点。

### 12.5 颜色与巡线结果（COLOR / LINE）

#### 12.5.1 `RPT_DETECT_COLOR (0x75)`

`RPT_DETECT_COLOR(0x75)` 采用“分组旋转框”格式：

```text
[group_count:u8] +
group_count * (
  [color_name:string_u8][blob_count:u8] +
  blob_count * ([center_x:u16][center_y:u16][w:u16][h:u16][angle:s16])
)
```

使用场景：

| 模式/场景 | 完整模式（`0x75`） | 简化模式 |
| --- | --- | --- |
| SingleColorDetection | 固定 `group_count=1` | `0x79`（`extra=color_name`） |
| MultiColorDetection | `group_count=有效颜色组数` | `0x79`（`extra=color_name`） |
| ColorTracking | 固定 `group_count=1`（当前学习名称） | `0x79`（`extra=color_name`） |

说明：

- `RPT_DETECT_COLOR(0x75)` 只返回颜色名、旋转框几何和角度，不包含 `score`。
- 其中 `center_x/center_y` 为旋转框中心点，`w/h` 为旋转框宽高，`angle` 为当前色块角度（度，和设备端颜色模式显示语义一致）。
- 简化模式 `0x79` 仅返回色块中心点和 `color_name`，不返回 `w/h`、`angle` 或 `score`。
- 以上“完整模式 `0x75`”指存在有效色块结果时；无结果时回退发送 `0x72` 空框（`count=0`）。

#### 12.5.2 `RPT_DETECT_LINE (0x76)`

Payload：

```text
[blob_count:u8][color_name:string_u8][center_pos:s16][angle:s16] +
blob_count * ([index:u8][center_x:u16][center_y:u16][w:u16][h:u16])
```

LineDetection 简化模式使用 `0x79`：

```text
[count:u8] +
count * (
  [center_x:u16=center_pos][center_y:u16=0][extra_count:u8=1] +
  [type=2][angle:s16]
)
```

说明：LineDetection 完整模式在无结果（`line_detected=0` 且 `blob_count=0`）时，回退发送 `0x72` 空框。

- `blob_count` 后每个 ROI 色块中的 `center_x/center_y` 也表示该 ROI 内命中色块的中心点，`w/h` 为该色块宽高。
- 巡线结果只返回 `center_pos`、`angle` 和各 ROI 色块几何，不返回 `score`。

### 12.6 关键点结果（KEYPOINT / HAND_KP / FACE_KP）

#### 12.6.1 `RPT_DETECT_KEYPOINT (0x77)` 人体关键点

Payload：

```text
[count:u8] +
count * (
  34 * [s16] +
  [id:string_u8] +
  [score:s16]
)
```

34 个 `s16` 表示 17 个 `(x,y)`，顺序（COCO-17）：

| 点序号 | 含义 |
| --- | --- |
| 0 | nose |
| 1 | left_eye |
| 2 | right_eye |
| 3 | left_ear |
| 4 | right_ear |
| 5 | left_shoulder |
| 6 | right_shoulder |
| 7 | left_elbow |
| 8 | right_elbow |
| 9 | left_wrist |
| 10 | right_wrist |
| 11 | left_hip |
| 12 | right_hip |
| 13 | left_knee |
| 14 | right_knee |
| 15 | left_ankle |
| 16 | right_ankle |

`PersonKeypointDetect(mode=8)` 忽略 `simple_result`；有关键点结果时上报 `0x77`，无结果时回退上报 `0x72` 空框。

其中 `id` 为匹配到的人体关键点模板名称；未命中模板时当前实现返回 `unknown`。`score` 为匹配分数百分比整数（`0..100`）。

#### 12.6.2 `RPT_DETECT_HAND_KP (0x78)` 手掌关键点

Payload：

```text
[count:u8] +
count * (
  [center_x:u16][center_y:u16][w:u16][h:u16] +
  42 * [s16] +
  [id:string_u8] +
  [score:s16]
)
```

42 个 `s16` 表示 21 个 `(x,y)`，顺序：

| 点序号 | 含义 |
| --- | --- |
| 0 | wrist |
| 1 | thumb_cmc |
| 2 | thumb_mcp |
| 3 | thumb_ip |
| 4 | thumb_tip |
| 5 | index_mcp |
| 6 | index_pip |
| 7 | index_dip |
| 8 | index_tip |
| 9 | middle_mcp |
| 10 | middle_pip |
| 11 | middle_dip |
| 12 | middle_tip |
| 13 | ring_mcp |
| 14 | ring_pip |
| 15 | ring_dip |
| 16 | ring_tip |
| 17 | little_mcp |
| 18 | little_pip |
| 19 | little_dip |
| 20 | little_tip |

模式约束：

- `HandKeyPointDetection(mode=11)` 且 `CMD_HAND_DETECT_ONLY=0`：完整模式使用 `0x78`
- `CMD_HAND_DETECT_ONLY=1`：完整模式回退到 `0x72`（仅手框）
- `CMD_HAND_DETECT_ONLY=0` 且无关键点结果时：完整模式回退到 `0x72` 空框
- 简化模式统一使用 `0x79`

手框字段说明：

- `center_x/center_y`：手掌检测框中心点。
- `w/h`：手掌检测框宽高。

其中 `id` 为匹配到的手掌关键点模板名称；未命中模板时当前实现返回 `unknown`。`score` 为匹配分数百分比整数（`0..100`）。

#### 12.6.3 `RPT_DETECT_FACE_KP (0x7A)` 人脸五点

Payload：

```text
[count:u8] + count * (10 * [s16])
```

10 个 `s16` 表示 5 个 `(x,y)`，顺序：

| 点序号 | 含义 |
| --- | --- |
| 0 | left_eye |
| 1 | right_eye |
| 2 | nose |
| 3 | left_mouth |
| 4 | right_mouth |

`RPT_DETECT_FACE_KP(0x7A)` 使用上述固定人脸五点结构。

### 12.7 按应用的上报类型速查

说明：

- 完整模式：`simple_result=0`
- 简化模式：`simple_result=1`

| 模式 | App | 完整模式 | 简化模式 |
| ---: | --- | --- | --- |
| 1 | FaceDetection | `0x72` | `0x79` |
| 2 | FaceLandmark | `0x72`（可带关键点 `extra`） | `0x79` |
| 3 | FacePose | `0x72` | `0x79` |
| 4 | FaceRecognition | `0x72` | `0x79` |
| 5 | FaceParse | `0x72` | `0x79` |
| 6 | FaceMesh | `0x72`（可带关键点 `extra`） | `0x79` |
| 7 | PersonDetection | `0x72` | `0x79` |
| 8 | PersonKeypointDetect | `0x77`（空结果回退 `0x72`） | `0x77`（空结果回退 `0x72`） |
| 9 | HandDetection | `0x72` | `0x79` |
| 10 | HandRecognition | `0x72` | `0x79` |
| 11 | HandKeyPointDetection | `0x78`（空结果回退 `0x72`）或 `0x72`（detect_only） | `0x79` |
| 12 | HandGesture | `0x72`（手势）或 `0x78`（关键点链路） | `0x79` |
| 13 | FaceLiveness | `0x72` | `0x79` |
| 14 | FalldownDetection | `0x72` | `0x79` |
| 15 | EyeGaze | `0x72` | `0x79` |
| 16 | ObjectTrack | `0x72` | `0x79` |
| 17 | GarbageClassification | `0x7B` | `0x79` |
| 18 | DynamicGesture | `0x73` | `0x73` |
| 19 | TrafficDetection | `0x72` | `0x79` |
| 20 | AiLLM_Mode | 无检测 `RPT` | 无检测 `RPT` |
| 21 | SingleColorDetection | `0x75`（空结果回退 `0x72`） | `0x79` |
| 22 | MultiColorDetection | `0x75`（空结果回退 `0x72`） | `0x79` |
| 23 | LineDetection | `0x76`（空结果回退 `0x72`） | `0x79` |
| 24 | ColorTracking | `0x75`（空结果回退 `0x72`） | `0x79` |
| 25 | OCRDetection | `0x7B` | `0x79` |
| 26 | OCRRecognition | `0x74` | `0x79` |
| 27 | LicencePlateDetection | `0x72` | `0x79` |
| 28 | LicencePlateRecognition | `0x74`（空结果回退 `0x72`） | `0x79` |
| 29 | ObjectDetection | 见 `12.8` | 见 `12.8` |
| 30 | Segmentation | `0x72` 或 `0x79`（取决于 `simple_result`） | `0x79` |
| 31 | SelfLearning | `0x72` | `0x73` |
| 32 | ApriltagDiscern | `0x7B` | `0x79` |
| 33 | DMCodeDiscern | `0x7B` | `0x79` |
| 34 | QRCodeDiscern | `0x72` | `0x73` |
| 35 | BarCodeDiscern | `0x72` | `0x73` |
| 36 | CustomDetection | 见 `12.9` | 见 `12.9` |

补充：上表为正常检测链路；异常分支还可能上报 `0x71`：

- `ObjectTrack(mode=16)` 发生跟踪失败时上报 `RPT_ERROR(0x71)`
- `OCRRecognition(mode=26)` 在 `count=0` 且存在内部错误信息时上报 `RPT_ERROR(0x71)`
- `QRCodeDiscern(mode=34)` / `BarCodeDiscern(mode=35)` 完整模式使用 `0x72`，框 extra 为 payload；简化模式使用 `0x73` 字符串列表。

### 12.8 `ObjectDetection(mode=29)` 物体检测

运行链路：

- `CMD_SET_MODE(29)` 启用 `FT_OBJECT_MODE_DETECT`
- 检测模型：`/sharefs/hiwonder/resources/kmodel/builtin/vision/yolo11s_320.kmodel`
- 标签来源：内置目标标签数组 `g_object_labels`

上报格式：

| `simple_result` | 上报码 | Payload 语义 |
| ---: | --- | --- |
| `0` | `0x72 RPT_DETECT_BBOX` | `[count] + count * ([center_x][center_y][w][h][extra_count] + [label?] + [score_pct])` |
| `1` | `0x79 RPT_DETECT_CENTER` | `[count] + count * ([center_x][center_y][extra_count] + [label?])` |

说明：

- `label` 非空时携带 `[type=1][label:string_u8]`；为空时省略。
- `score_pct` 使用 `[type=3][score_pct:s16]`，为百分比整数，当前完整模式始终携带。
- `simple_result=1` 不携带分数。
- `CMD_OBJECT_SET_MODE(0x5B)` 选择对象运行链路：`0=detect`，`2=seg`。`ObjectDetection(mode=29)` 使用 `detect`，`Segmentation(mode=30)` 使用 `seg`。

`ObjectDetection` 标签集合：

| 序号 | 协议标签 |
| ---: | --- |
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 4 | airplane |
| 5 | bus |
| 6 | train |
| 7 | truck |
| 8 | boat |
| 9 | traffic light |
| 10 | fire hydrant |
| 11 | stop sign |
| 12 | parking meter |
| 13 | bench |
| 14 | bird |
| 15 | cat |
| 16 | dog |
| 17 | horse |
| 18 | sheep |
| 19 | cow |
| 20 | elephant |
| 21 | bear |
| 22 | zebra |
| 23 | giraffe |
| 24 | backpack |
| 25 | umbrella |
| 26 | handbag |
| 27 | tie |
| 28 | suitcase |
| 29 | frisbee |
| 30 | skis |
| 31 | snowboard |
| 32 | sports ball |
| 33 | kite |
| 34 | baseball bat |
| 35 | baseball glove |
| 36 | skateboard |
| 37 | surfboard |
| 38 | tennis racket |
| 39 | bottle |
| 40 | wine glass |
| 41 | cup |
| 42 | fork |
| 43 | knife |
| 44 | spoon |
| 45 | bowl |
| 46 | banana |
| 47 | apple |
| 48 | sandwich |
| 49 | orange |
| 50 | broccoli |
| 51 | carrot |
| 52 | hot dog |
| 53 | pizza |
| 54 | donut |
| 55 | cake |
| 56 | chair |
| 57 | couch |
| 58 | potted plant |
| 59 | bed |
| 60 | dining table |
| 61 | toilet |
| 62 | tv |
| 63 | laptop |
| 64 | mouse |
| 65 | remote |
| 66 | keyboard |
| 67 | cell phone |
| 68 | microwave |
| 69 | oven |
| 70 | toaster |
| 71 | sink |
| 72 | refrigerator |
| 73 | book |
| 74 | clock |
| 75 | vase |
| 76 | scissors |
| 77 | teddy bear |
| 78 | hair drier |
| 79 | toothbrush |

### 12.9 `CustomDetection(mode=36)` 自定义识别

运行链路：

- `CMD_SET_MODE(36)` 启用自定义识别
- `CMD_CUSTOM_SET_MODEL(0x5C)` 选择 `custom_detect_models` 中的模型条目
- 模型配置从 `config.json` 读取，按以下路径顺序查找：
  - `/sharefs/hiwonder_runtime/config.json`
  - `/sdcard/hiwonder/config.json`
  - `A:/hiwonder/config.json`

#### 12.9.1 支持的 YOLO family 与任务

`family` 用于选择 YOLO 输出解码布局：

| family | 可写值 |
| --- | --- |
| YOLOv5 | `5` / `v5` / `yolov5` / `yolo5` |
| YOLOv8 | `8` / `v8` / `yolov8` / `yolo8` |
| YOLO11 | `11` / `v11` / `yolo11` / `yolov11` |

`task` 用于选择结果类型：

| task | 可写值 | 完整模式上报 | 简化模式上报 |
| --- | --- | --- | --- |
| 水平框目标检测 | `detect` / `det` | `0x72 RPT_DETECT_BBOX`，extra 为 `[label][score_pct]` | `0x79 RPT_DETECT_CENTER`，extra 为 `[label]` |
| 旋转框目标检测 | `obb` | `0x7B RPT_DETECT_QUAD`，extra 为 `[label][score_pct]` | `0x79 RPT_DETECT_CENTER`，extra 为 `[label]` |
| 图像分类 | `cls` / `classify` / `classification` | `0x79 RPT_DETECT_CENTER`，extra 为 `[label]` | `0x79 RPT_DETECT_CENTER`，extra 为 `[label]` |

说明：

- `labels` 顺序必须与模型输出类别 ID 一致，协议返回的 `label` 即对应字符串。
- `score_pct` 为百分比整数，编码为 `[type=3][score_pct:s16]`。
- `task=detect` 时，完整模式 `0x72` 中的 `center_x/center_y` 表示目标中心点，`w/h` 为目标框宽高。
- `task=obb` 时，完整模式 `0x7B` 也只在 `extra` 中携带 `[label][score_pct]`。
- `cls` 结果以画面中心点作为 `center_x/center_y`，最多上报通过阈值的前 5 个分类标签，并受 `max_results` 限制。
- `cls` 结果当前协议不返回分类分数，`CENTER extra` 中只有 `label`。

#### 12.9.2 `config.json` 写法

根对象中添加 `custom_detect_models` 数组，每个有效条目对应一个可选择模型。`CMD_CUSTOM_SET_MODEL(0x5C)` 的 `model_index` 按有效条目顺序从 `0` 开始编号，当前外部命令范围为 `0..15`。

示例：

```json
{
  "custom_detect_models": [
    {
      "model": "my_yolov8_detect_320.kmodel",
      "family": "yolov8",
      "task": "detect",
      "labels": ["person", "car", "dog"],
      "conf_thresh": 0.30,
      "nms_thresh": 0.45,
      "max_results": 20
    },
    {
      "model": "my_yolov8_obb_320.kmodel",
      "family": "yolov8",
      "task": "obb",
      "labels": ["box", "card", "bottle"],
      "conf_thresh": 0.50,
      "nms_thresh": 0.60,
      "max_results": 20
    },
    {
      "model": "my_yolo11_cls_224.kmodel",
      "family": "yolo11",
      "task": "cls",
      "labels": ["normal", "defect"],
      "conf_thresh": 0.70,
      "max_results": 1
    }
  ]
}
```

字段说明：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `model` | 是 | 模型文件名或绝对路径；相对写法只能写文件名，不能包含 `/` 或 `\\` |
| `family` | 否 | 默认 `yolov8`；也可写作 `yolo_family` / `model_family` |
| `task` | 否 | 默认 `detect`；也可写作 `yolo_task` / `model_task` |
| `labels` | 是 | 字符串数组，类别顺序必须与模型输出一致 |
| `conf_thresh` | 否 | 置信度阈值，默认 `0.30`；`cls` 上报阈值不低于 `0.70` |
| `nms_thresh` | 否 | NMS 阈值，默认 `0.45`；`cls` 不使用该字段 |
| `max_results` | 否 | 单帧最大结果数，默认 `50`，最大按 `50` 裁剪 |
| `obb` | 否 | 未填写 `task` 时，`true` 等价于 `task:"obb"` |

模型文件查找规则：

- `model` 为绝对路径时直接使用该路径。
- `model` 为文件名时，优先查找 `/sharefs/hiwonder/resources/kmodel/custom/<model>`。
- 未在自定义模型目录找到时，会继续查找系统内置模型目录；交付自定义模型时建议放入自定义模型目录。

配置约束：

- `custom_detect_models` 最多通过外部命令选择前 16 个有效条目。
- `labels` 最多 128 项；单个标签最长 31 字节；标签文本总长度不超过 1023 字节。
- 模型输入尺寸从 kmodel 读取，配置文件不需要填写宽高。

## 13. 模式索引表

`CMD_SET_MODE` payload 为 `mode_index:u8`：

| 索引 | 名称 |
| ---: | --- |
| 0 | Empty |
| 1 | FaceDetection |
| 2 | FaceLandmark |
| 3 | FacePose |
| 4 | FaceRecognition |
| 5 | FaceParse |
| 6 | FaceMesh |
| 7 | PersonDetection |
| 8 | PersonKeypointDetect |
| 9 | HandDetection |
| 10 | HandRecognition |
| 11 | HandKeyPointDetection |
| 12 | HandGesture |
| 13 | FaceLiveness |
| 14 | FalldownDetection |
| 15 | EyeGaze |
| 16 | ObjectTrack |
| 17 | GarbageClassification |
| 18 | DynamicGesture |
| 19 | TrafficDetection |
| 20 | AiLLM_Mode |
| 21 | SingleColorDetection |
| 22 | MultiColorDetection |
| 23 | LineDetection |
| 24 | ColorTracking |
| 25 | OCRDetection |
| 26 | OCRRecognition |
| 27 | LicencePlateDetection |
| 28 | LicencePlateRecognition |
| 29 | ObjectDetection |
| 30 | Segmentation |
| 31 | SelfLearning |
| 32 | ApriltagDiscern |
| 33 | DMCodeDiscern |
| 34 | QRCodeDiscern |
| 35 | BarCodeDiscern |
| 36 | CustomDetection |

## 14. 功能码规范（CMD）

### 14.1 系统控制

| Func | 名称 | Payload | 成功 `extra` | 说明 |
| --- | --- | --- | --- | --- |
| `0x01` | CMD_SET_MODE | `[mode:u8]` | `01` | 切换模式，不等于自动启动 |
| `0x02` | CMD_SET_VOLUME | `[volume:u8]` | `01` | 范围 `0..100` |
| `0x03` | CMD_SET_WIFI | `[ssid:string_u8][password:string_u8]` | `01` | 最终结果见 `0x6E` |
| `0x04` | CMD_REQUEST_STATUS | 空（建议） | 空 | 附加触发一次状态心跳 |
| `0x05` | CMD_CLEAR_MEMORY | 空 | `01` | 清结果状态，不改 mode/run |
| `0x06` | CMD_GET_PROTOCOL_INFO | 空 | 8 字节 | `major, minor, caps, max_frame_len` |

### 14.2 检测参数

| Func | 名称 | Payload | 说明 |
| --- | --- | --- | --- |
| `0x10` | CMD_SET_CONF_THRESH | `[threshold:u8]` | `0..100` |
| `0x11` | CMD_SET_NMS_THRESH | `[threshold:u8]` | `0..100` |
| `0x12` | CMD_SEG_SET_MASK_THRESH | `[threshold:u8]` | `0..100` |
| `0x13` | CMD_SET_SIMPLE_RESULT | `[enabled:u8]` | `0/1` |
| `0x14` | CMD_DISABLE_RUN | `[disable:u8]` | `0=运行, 1=停止` |

### 14.3 人脸 / 人体关键点 / 手掌关键点

| Func | 名称 | Payload |
| --- | --- | --- |
| `0x20` | CMD_FACE_LEARN | `[name:string_u8]` |
| `0x21` | CMD_FACE_DELETE | `[name:string_u8]` |
| `0x22` | CMD_FACE_RENAME | `[old:string_u8][new:string_u8]` |
| `0x23` | CMD_FACE_SET_RECOG_CONF | `[threshold:u8]` |
| `0x24` | CMD_FACE_HIGH_PRECISION | `[enabled:u8]` |
| `0x25` | CMD_FACE_ENABLE_KEYPOINT | `[enabled:u8]` |
| `0x26` | CMD_FACE_DETECT_ONLY | `[enabled:u8]` |
| `0x27` | CMD_FACE_ENHANCE_LEARN | `[name:string_u8]` |
| `0x28` | CMD_FACE_LEARN_AT_POINT | `[x:u32][y:u32][name:string_u8]` |
| `0x29` | CMD_FACE_SET_POSE_THRESH | `[roll:u8][pitch:u8][yaw:u8]` |
| `0x2A` | CMD_PERSON_KP_LEARN | `[name:string_u8]` |
| `0x2B` | CMD_PERSON_KP_DELETE | `[name:string_u8]` |
| `0x2C` | CMD_PERSON_KP_RENAME | `[old:string_u8][new:string_u8]` |
| `0x2D` | CMD_PERSON_KP_ENHANCE_LEARN | `[name:string_u8]` |
| `0x2E` | CMD_HAND_KP_LEARN | `[name:string_u8]` |
| `0x2F` | CMD_HAND_KP_DELETE | `[name:string_u8]` |
| `0x30` | CMD_HAND_KP_RENAME | `[old:string_u8][new:string_u8]` |
| `0x31` | CMD_HAND_KP_ENHANCE_LEARN | `[name:string_u8]` |
| `0x32` | CMD_HAND_DETECT_ONLY | `[enabled:u8]` |

约束与行为：

- `0x20/0x21/0x22/0x27/0x28` 支持模式：`FaceRecognition(4)`、`FacePose(3)`、`EyeGaze(15)`
- `0x23`、`0x24` 支持模式：`FaceDetection(1)`、`FaceLandmark(2)`、`FacePose(3)`、`FaceRecognition(4)`、`FaceParse(5)`、`FaceMesh(6)`、`FaceLiveness(13)`、`EyeGaze(15)`
- `0x25`、`0x26` 用于 `FaceRecognition(4)` 子模式控制
- `0x32` 在 `HandDetection(9)`、`HandKeyPointDetection(11)`、`HandGesture(12)` 有效
- `0x29` 在 `FacePose(3)` 与 `EyeGaze(15)` 有效，阈值范围 `1..180`
- `0x29` 默认值：`roll=18, pitch=18, yaw=24`
- 学习类命令在 `run=0` 时返回 `ERR_EXEC_FAIL`：
  - `0x20`、`0x27`、`0x28`、`0x2A`、`0x2D`、`0x2E`、`0x31`
- 人脸/人体关键点/手掌关键点学习类命令先回原命令 `RSP`，随后通过 `RSP func=0x6E` 给最终结果

### 14.4 颜色识别 / 颜色学习 / 巡线

| Func | 名称 | Payload |
| --- | --- | --- |
| `0x40` | CMD_COLOR_SET_TARGET | `[name:string_u8]` |
| `0x41` | CMD_COLOR_SET_THRESH | `[count:u8] + count * ([name:string_u8][lab6])` |
| `0x42` | CMD_COLOR_GET_THRESH | `[name:string_u8]` |
| `0x43` | CMD_COLOR_SET_FILTER | `[number:u8][min_area:u32][max_area:u32]` |
| `0x44` | CMD_COLOR_SET_MIN_AREA | `[min_area:u32]` |
| `0x45` | CMD_MULTI_COLOR_SET_LIST | `[count:u8][name:string_u8]...` |
| `0x46` | CMD_LINE_SET_ROI | 15 字节 ROI |
| `0x47` | CMD_COLOR_LEARNING_SET_POINT | `[x:u16][y:u16][name:string_u8]` |
| `0x48` | CMD_COLOR_LEARNING_SAVE | `[name:string_u8]` |
| `0x49` | CMD_COLOR_LEARNING_RENAME | `[old:string_u8][new:string_u8]` |
| `0x4A` | CMD_COLOR_LEARNING_DELETE | `[name:string_u8]` |

`CMD_COLOR_GET_THRESH` 成功时，`RSP extra` 固定返回 6 字节阈值。

模式映射：

- `SingleColorDetection(21)`：`0x40/0x41/0x42/0x43/0x44`
- `MultiColorDetection(22)`：`0x41/0x42/0x43/0x44/0x45`
- `LineDetection(23)`：`0x40/0x41/0x42/0x43/0x44/0x46`
- `ColorTracking(24)`：`0x40/0x42/0x43/0x44/0x47/0x48/0x49/0x4A`

语义说明：

- `0x40`：在 `SingleColorDetection(21)`、`LineDetection(23)`、`ColorTracking(24)` 下设置当前目标颜色/选中颜色项名称。
- `0x41`：仅在 `SingleColorDetection(21)`、`MultiColorDetection(22)`、`LineDetection(23)` 有效；每个阈值项的 `lab6` 取值约束见 `9.2`。
- `0x43`：`number=0` 时允许 `min_area/max_area=0`；`number!=0` 时要求 `min_area>=1`、`max_area>=1` 且 `max_area >= min_area`。
- `0x46`：ROI 使用百分比编码，字段约束见 `9.3`。
- `0x47`：按坐标设置/学习目标色；仅在 `ColorTracking(24)` 有效，坐标范围 `x=0..319`、`y=0..239`。
- `0x48`：把当前学习结果保存为指定名称。
- `0x49` / `0x4A`：重命名 / 删除颜色学习名称，并同步设备内部当前选中态。

### 14.5 自学习 / 目标跟踪 / 动态手势 / 自定义模型 / 物体子模式

| Func | 名称 | Payload | 说明 |
| --- | --- | --- | --- |
| `0x50` | CMD_NANOTRACK_SET_RECT | `[x:u16][y:u16][w:u16][h:u16]` | `w/h` 约束 `36..240` |
| `0x51` | CMD_NANOTRACK_STOP | `[stop:u8]` | 停止跟踪 |
| `0x52` | CMD_GESTURE_SET_FRAME | `[frame:u16]` | `1..120` |
| `0x53` | CMD_DGESTURE_CTRL | `[action:u8][name?][name2?]` | 动态手势控制 |
| `0x54` | CMD_DGESTURE_ENHANCE_SAVE | `[name:string_u8]` | 增强保存 |
| `0x55` | CMD_SELFLEARN_SET_NAME | `[name:string_u8]` | 启动一次自学习采样 |
| `0x56` | CMD_SELFLEARN_SET_RECT | `[x:u16][y:u16][w:u16][h:u16]` | 最小 `24x24` |
| `0x57` | CMD_SELFLEARN_SET_FRAME | `[frame:u16]` | `1..120` |
| `0x58` | CMD_SELFLEARN_SET_FEATURES | `[features:u8]` | `1..16` |
| `0x59` | CMD_SELFLEARN_DELETE | `[name:string_u8]` | 删除样本 |
| `0x5A` | CMD_SELFLEARN_RENAME | `[old:string_u8][new:string_u8]` | 重命名样本 |
| `0x5B` | CMD_OBJECT_SET_MODE | `[mode:u8]` | `0=detect,2=seg` |
| `0x5C` | CMD_CUSTOM_SET_MODEL | `[model_index:u32]` | `custom_detect_models` 有效条目索引，范围 `0..15` |

`CMD_DGESTURE_CTRL action`：

| 值 | 含义 |
| --- | --- |
| `1` | RECORD_START |
| `2` | RECORD_STOP |
| `3` | SAVE |
| `4` | DELETE |
| `5` | RENAME |
| `6` | SAVE_APPEND |
| `7` | SAVE_APPEND_DROP_OLDEST |

约束与行为：

- `CMD_DGESTURE_CTRL(0x53)` 要求模式 `DynamicGesture(18)` 且 `run=1`
- `CMD_DGESTURE_ENHANCE_SAVE(0x54)` 与 `0x53` 同约束（模式 `18` 且 `run=1`）
- `run=0` 下发送任何 `action` 返回 `ERR_EXEC_FAIL`
- `CMD_OBJECT_SET_MODE(0x5B)` 在 `ObjectDetection(29)` / `Segmentation(30)` 下生效：`ObjectDetection(29)` 使用 `0=detect`，`Segmentation(30)` 使用 `2=seg`
- `CMD_CUSTOM_SET_MODEL(0x5C)` 仅在 `CustomDetection(36)` 有效，索引含义见 `12.9`
- 自学习“清空”仅板端内部命令，当前外部协议未暴露独立功能码

### 14.6 AI / 语音 / MCP / 大模型

| Func | 名称 | Payload |
| --- | --- | --- |
| `0x60` | CMD_SET_LLM_KEY | `[string_u8]` |
| `0x61` | CMD_SET_TTS_VOICE | `[model:string_u8][voice:string_u8]` |
| `0x62` | CMD_SET_ASR_LANG | `[string_u8]` |
| `0x63` | CMD_SET_THINKING | `[enabled:u8]` |
| `0x64` | CMD_SET_SEARCH | `[enabled:u8]` |
| `0x65` | CMD_SET_START_SILENCE | `[ms:u16]` |
| `0x66` | CMD_SET_END_SILENCE | `[ms:u16]` |
| `0x67` | CMD_SET_PROMPT | `[string_u8]` |
| `0x68` | CMD_ASR | `[start:u8]` |
| `0x69` | CMD_TTS | `[string_u8]` |
| `0x6A` | CMD_LLM_CHAT | `[string_u8]` |
| `0x6B` | CMD_VLM_CHAT | `[string_u8]` |
| `0x6C` | CMD_SET_MCP_TOOLS | `data_pack(obj)` |
| `0x6D` | CMD_RESULT_RETURN | `data_pack(obj)` |
| `0x6F` | CMD_SET_LLM_MODEL | `[string_u8]` |
| `0x70` | CMD_SET_VLM_MODEL | `[string_u8]` |
| `0x71` | CMD_SET_LLM_BASE_URL | `[string_u8]` |
| `0x72` | CMD_SET_VLM_BASE_URL | `[string_u8]` |
| `0x73` | CMD_SET_SPEECH_URL | `[string_u8]` |

约束与行为：

- `CMD_SET_START_SILENCE` / `CMD_SET_END_SILENCE` 范围：`0..30000` ms
- `CMD_ASR`、`CMD_TTS`、`CMD_LLM_CHAT`、`CMD_VLM_CHAT` 为异步命令
- `CMD_SET_MCP_TOOLS(0x6C)` 注册的是需要外部主机执行的 MCP 工具；当工具被调用时，从机通过 `RSP func=0x6D` 把 `{tool_name: args}` 转发给主机
- 主机收到外部 MCP 工具调用并执行完后，应再主动发送 `CMD_RESULT_RETURN(0x6D)`，payload 为 `data_pack(result)`，把执行结果回填给从机
- 固件内置 MCP 控制项由设备本地消费，不转发给主机；例如 `MCP_SET_VOLUME` 会直接调节本机音量，`MCP_CAMERA_TAKE_PHOTO` / `MCP_CAMERA_VISION` / `MCP_CAMERA_SHOW` / `MCP_CAMERA_HIDE` 会直接驱动本机相机相关流程
- 主机 `MUST NOT` 主动发送 `0x6E CMD_EMPTY_RETURN`

### 14.7 媒体 / 相机

| Func | 名称 | Payload | 说明 |
| --- | --- | --- | --- |
| `0x81` | CMD_MEDIA_CAMERA_SNAPSHOT | 空 | 触发拍照并保存 |
| `0x82` | CMD_MEDIA_SET_PHOTO_PREFIX | `[prefix:string_u8]` | 允许空字符串 |
| `0x83` | CMD_MEDIA_DELETE_PHOTO | `[name:string_u8]` | 支持基础名或完整名 |
| `0x85` | CMD_MEDIA_ENTER_CAMERA_APP | 空 | 切换至媒体相机 app |
| `0x86` | CMD_MEDIA_SET_PHOTO_START | `[start:u32]` | 设置前缀序号起点 |

其他说明：

- `CMD_MEDIA_SET_PHOTO_PREFIX` 仅更新前缀配置，不立即拍照
- 前缀为空时，后续拍照回退为时间戳命名
- `CMD_MEDIA_ENTER_CAMERA_APP` 仅切页面，不直接触发拍照

## 15. `CMD_GET_PROTOCOL_INFO (0x06)` 返回格式

成功时 `RSP extra` 固定 8 字节：

```text
[major:u8][minor:u8][capability_flags:u32_be][max_frame_len:u16_be]
```

当前 capability flags：

| Bit | 值 | 含义 |
| --- | --- | --- |
| bit0 | `0x00000001` | 支持 `RSP` 帧 |
| bit1 | `0x00000002` | I2C mailbox v2 |
| bit2 | `0x00000004` | 分包重组检查序号 |
| bit3 | `0x00000008` | 支持 `CMD_GET_PROTOCOL_INFO` |
| bit4 | `0x00000010` | 心跳不携带命令结果 |
| bit5 | `0x00000020` | `RSP/RPT_ERROR` 携带 `module+subcode` 错误明细 |
| bit6 | `0x00000040` | 心跳为 idle-only keepalive（空闲保活） |

## 16. 错误码

### 16.1 错误字段模型

协议采用三段式错误模型：

```text
[err_code:u8][err_module:u8][err_subcode:u16_be]
```

- `err_code`：主错误码（错误类型）
- `err_module`：错误来源模块（责任归属）
- `err_subcode`：细分场景（定位点）

该模型统一用于：

- `RSP` 响应帧（第 10 章）
- `RPT_ERROR(0x71)` 上报帧（第 11 章）

### 16.2 主错误码表（err_code）

| 值 | 名称 | 错误类 | 默认模块 | 重试建议 | 典型触发 |
| --- | --- | --- | --- | --- | --- |
| `0x00` | `ERR_OK` | `ok` | `none` | 否 | 命令执行成功 |
| `0x01` | `ERR_UNKNOWN_CMD` | `command` | `command` | 否 | 功能码未实现/不识别 |
| `0x02` | `ERR_INVALID_MODE` | `command` | `command` | 否 | 当前 `mode` 不支持该命令 |
| `0x03` | `ERR_INVALID_PARAM` | `command` | `command` | 否 | 参数越界、非法字符串、非法 flags 组合 |
| `0x04` | `ERR_DATA_LEN` | `command` | `command` | 否 | payload 长度不符、字段不完整 |
| `0x05` | `ERR_BUSY` | `runtime` | `runtime` | 是 | 异步任务占用中（如语音异步未完成） |
| `0x06` | `ERR_NOT_READY` | `runtime` | `runtime` | 是 | 系统未就绪、IPC 未连通、服务未准备好 |
| `0x07` | `ERR_BUFFER_FULL` | `runtime` | `runtime` | 是 | 响应队列/发送窗口/编码缓存满 |
| `0x08` | `ERR_EXEC_FAIL` | `runtime` | `runtime` | 是 | 业务调用失败（算法/文件/IPC 执行失败） |
| `0x09` | `ERR_FRAME_INVALID` | `frame` | `protocol` | 是 | 帧头/帧结构非法 |
| `0x0A` | `ERR_XOR_FAIL` | `frame` | `protocol` | 是 | 帧 XOR 校验失败 |
| `0x0B` | `ERR_SEQ_MISMATCH` | `frame` | `protocol` | 是 | 分包序号/func/txn 不连续或不匹配 |
| `0x0C` | `ERR_REASSEMBLE_FAIL` | `frame` | `protocol` | 是 | 分包重组缓存溢出/重组失败 |
| `0x0D` | `ERR_CALLBACK_FAIL` | `runtime` | `runtime` | 是 | 从机内部命令回调未注册/不可用 |
| `0x0E` | `ERR_UART_WRITE` | `transport` | `transport` | 是 | UART 写失败 |
| `0x0F` | `ERR_UART_READ` | `transport` | `transport` | 是 | UART 读失败 |

### 16.3 错误模块表（err_module）

| 值 | 名称 | 说明 |
| --- | --- | --- |
| `0x00` | `none` | 成功或未细分 |
| `0x01` | `protocol` | 协议层、编解码、分包重组 |
| `0x02` | `transport` | UART/I2C 传输层 |
| `0x03` | `command` | 命令语义与模式约束 |
| `0x04` | `runtime` | 系统运行态/启动阶段 |
| `0x05` | `ipc` | 大小核 IPC、外部服务转发 |
| `0x06` | `report` | 结果上报构建/发布 |
| `0x07` | `speech` | 语音/LLM/VLM/MCP 异步链路 |
| `0x08` | `media` | 相机/相册媒体链路 |
| `0xFF` | `unknown` | 未知来源（保留） |

### 16.4 错误子码（err_subcode）规则

- `0x0000`：未细分
- `0x8000 | func_code`：按命令功能码映射的子码（命令处理失败默认形式）

当前已定义专项子码：

| 子码 | 名称 | 说明 |
| --- | --- | --- |
| `0x0101` | `WL_MCU_ERR_SUB_CMD_PRE_STARTUP` | 启动门控未打开即收到业务命令 |
| `0x0102` | `WL_MCU_ERR_SUB_CMD_PROTOCOL_INFO_BUILD` | `CMD_GET_PROTOCOL_INFO` 构建失败 |
| `0x0103` | `WL_MCU_ERR_SUB_CMD_QUEUE_FULL` | 命令响应队列满 |
| `0x0201` | `WL_MCU_ERR_SUB_IPC_WIFI_SEND` | Wi-Fi IPC 请求发送失败 |
| `0x0202` | `WL_MCU_ERR_SUB_IPC_SPEECH_SEND` | 语音 IPC 请求发送失败 |
| `0x0301` | `WL_MCU_ERR_SUB_ASYNC_TIMEOUT` | 异步命令等待超时 |
| `0x0401` | `WL_MCU_ERR_SUB_REPORT_OBJTRACK` | 跟踪结果上报失败 |
| `0x0402` | `WL_MCU_ERR_SUB_REPORT_OCR` | OCR 结果上报失败 |
| `0x0501` | `WL_MCU_ERR_SUB_UART_CALLBACK_MISSING` | UART 命令回调缺失 |
| `0x0502` | `WL_MCU_ERR_SUB_I2C_CALLBACK_MISSING` | I2C 命令回调缺失 |
| `0x0601` | `WL_MCU_ERR_SUB_REASSEMBLE_OVERFLOW` | 重组缓存溢出 |

### 16.5 使用约束

- 主机 `MUST` 按数值处理错误字段，不依赖文本描述。
- 主机 `SHOULD` 同时记录 `err_code + err_module + err_subcode`，用于日志检索与统计。
- `err_code!=0` 且 `err_subcode=0x0000` 表示仅返回到主码/模块粒度。
- 对 `frame/transport/runtime` 且“建议重试=是”的错误，可退避重试；对 `command` 类错误应先修正请求。

## 17. 主机实现要求

### 17.1 最低能力（MUST）

主机实现至少应具备：

- 帧解析与 XOR 校验
- 分包发送与重组
- Txn 生命周期管理
- `RSP` / `RPT` 分流
- I2C mailbox v2 状态机
- CompactCodec 与 `data_pack` 解码

### 17.2 推荐启动流程（SHOULD）

1. 上电后执行 `CMD_CLEAR_MEMORY`
2. 执行 `CMD_GET_PROTOCOL_INFO`
3. 根据 `max_frame_len` 配置分包阈值
4. 建立“任意有效协议帧 = transport alive”的活跃判定
5. 进入业务命令和上报轮询

### 17.3 禁止实现（MUST NOT）

- 依赖“心跳携带命令结果”
- 依赖“持续收到独立心跳”作为链路在线的唯一判据
- 主机命令使用 `Txn=0`
- I2C 下直接向偏移 `0` 写协议帧
- 把巡线 ROI 当像素发送
- 依据模式名硬编码结果解析分支

### 17.4 本地配置重载语义

以下资源按“进入对应 app 初始化时重载一次”生效：

- `color_config.json`
- `lab_data.json`
- 自学习特征库
- 人体关键点 / 手部关键点资料库

主机侧应按以下语义处理：

- 功能 `disabled -> enabled` 或重新进入 app，会触发下一次重载
- 同一活跃 session 内重复下发相同启用命令，不保证再次重载
- 若运行中修
改了配置文件，需先退出当前 app（或关闭功能）再重新进入

## 18. 交互示例

### 18.1 `CMD_SET_MODE(FaceRecognition)` 示例

假设：

- `mode=1`
- `txn=7`
- `seq=0`

命令帧：

```text
AA 55 00 01 00 01 07 01 06
```

成功响应：

```text
AA 55 00 05 40 01 07 00 00 00 00 01 42
```

解析：

- `Ctrl=0x40`：RSP
- `Func=0x01`
- `Txn=0x07`
- `Payload=00 00 00 00 01`
  - `err_code=0x00`
  - `err_module=0x00`
  - `err_subcode=0x0000`
  - `extra=0x01`

### 18.2 `CMD_GET_PROTOCOL_INFO` 成功响应示例

```text
Payload = 00 00 00 00 02 04 00 00 00 7F 01 00
```

解释：

- `00`：命令成功
- `00`：错误模块（none）
- `00 00`：错误子码
- `02`：协议主版本
- `06`：协议次版本
- `00 00 00 7F`：能力位
- `01 00`：当前最大帧长 `256`

## 19. 协议验收清单

建议在主机交付前逐项自检：

- 是否严格按 `func_code` 解析 `RPT`
- 是否完整实现 `Txn` 生命周期
- 是否完成 I2C mailbox v2 的 state/generation 管理
- 是否正确实现分包重组与序号校验
- 是否支持 `CompactCodec` 与 `data_pack`
- 是否覆盖异步命令两阶段返回处理
- 是否验证 `CMD_GET_PROTOCOL_INFO` 动态帧长能力
- 是否完整解析并记录 `err_code + err_module + err_subcode`
