# NexArm follower firmware: the real control path behind CMD 96/97

Status: research note, 2026-10-23. Sources: `firmware/esp32/follower/Nex_Arm/*`
(Arduino source), `firmware/esp32/leader/Nex_Arm/*`, and the embedded AT32
application image `at32_firmware.h` (v1.0.0, 21,688 bytes, no source). The
AT32 findings come from a Thumb-2 disassembly of that image with capstone; the
helper script is in the session scratchpad and is not committed. Addresses are
given so the reading can be checked. Everything marked (inferred) is not
proven by code alone.

## 1. Topology

```
PC ──USB (CH340, 1 Mbps)── ESP32 UART0 ──┐
                                         │ ESP32 "servo" UART = Serial1, GPIO16/17, 1 Mbps
                                         ▼
                      AT32F421 USART1 (PA9/PA10)  ◄── command link, ID 0xFF in / 0x5A out
                      AT32F421 USART2 (0x40004400) ◄─► HX servo bus, IDs 1..6, 0xFE broadcast
```

- The ESP32 never talks to a servo register itself in LeRobot mode. Every
  CMD 96/97/98 frame from the PC is re-framed with ID 0xFF and written to
  Serial1; the AT32 does the bus work and answers with ID 0xFF
  (`system_task_handle.cpp:99-134`, `bridge_at32_callback`).
- The ESP32 _does_ write raw servo frames on the same UART in ESP-NOW teach
  mode (`EspNow_Ctrl.cpp:76`, `sync_write_pos_speed` to ID 0xFE) and for torque
  and acceleration broadcasts (`set_servo_realtime_mode`). The AT32 application
  drops every frame whose ID is not 0xFF (main loop at `0x800561c`), so those
  raw frames can only reach the servos if the ESP32 UART is electrically on
  the servo bus as well (inferred). The `servo_uart_busy` mutex and the RX
  flushes before ESP-NOW start support this reading.
- The leader board has no AT32: its ESP32 reads the six servos directly, one
  register read per servo with an 800 µs gap (`leader/Nex_Arm.ino:60-75`).

## 2. ESP32 bridge mode (CMD 68)

- Entering bridge mode swaps the USB callback for `bridge_usb_callback` and the
  AT32 callback for `bridge_at32_callback`. Leaving it reboots the ESP32
  (`ESP.restart()` at `system_task_handle.cpp:2953`).
- In bridge mode, **every** PC command is forwarded to the AT32 with ID 0xFF,
  not only 96/97/98 (default branch at `:126-129`). So CMD 56, 59, 60, 71..80
  and 104 all reach the AT32 while the robot is in LeRobot use.
- Only AT32 replies with cmd 96, 97 or 98 are relayed back to the PC
  (`:87-95`). Replies to anything else (PID readback, overload readback, the
  10 ms position stream) are silently dropped on the ESP32. This is a few-line
  change in open source.
- The bridge loop does nothing else: parse USB, parse AT32, check the boot
  button. No periodic work, so ESP32 adds well under 1 ms (estimate).

## 3. AT32 application v1.0.0, disassembled

### 3.1 Dispatcher

`0x80041f8`: `subs r0,#6; cmp r0,#0x62; bhi default; tbh` — one jump table for
commands 6..104, indexed by cmd − 6. Frames with ID ≠ 0xFF never reach it.
Entries that matter here:

| CMD                | Handler                   | What it does                                                                                                                                                    |
| ------------------ | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 56 `SET_MOVE_ACC`  | `0x8004570`               | Stores **one byte** `args[0]` into `arm.move_acc` (`0x200000b0+0x39`). Nothing else. The speed bytes LeRobot sends are ignored.                                 |
| 59 `SET_PID_PARAM` | `0x80045dc`               | Writes 5 bytes at servo register 21: `{args[1], args[3], args[2], args[4], args[5]}` to servo `args[0]`.                                                        |
| 60 `GET_PID_PARAM` | `0x8004614`               | Reads 5 bytes at register 21 from servo `args[0]`; replies cmd 60 with `{id, b0, b2, b1, b3, b4}` (ID 0x5A).                                                    |
| 71 / 72            | `0x8004890` / `0x80048f4` | Read / write 3 bytes at register 34 (protection torque, protection time, overload torque).                                                                      |
| 75 / 76            | `0x8004988` / `0x80049ce` | Read / write the 16-bit register 48 (max torque).                                                                                                               |
| 57 / 58            | `0x8004584` / `0x800459c` | Write / read the 16-bit position offset at register 31.                                                                                                         |
| 96 `LR_READ_POS`   | `0x8004c78`               | Sets flag `0x200001ed = 1`, then **one sync read** of IDs 1..6 at register 56, length 8, and replies cmd 96 with six int16 positions.                           |
| 97 `LR_WRITE_POS`  | `0x8004cd6`               | Builds six records `{id, acc = 0, speed = 0, pos}` and issues **one sync write** at register 41 (length 7 per servo). No queue, no interpolation, no speed cap. |
| 98 `LR_TORQUE`     | `0x8004d22`               | Six individual torque-enable writes.                                                                                                                            |
| 104 `SERVO_STREAM` | `0x8004d4c`               | Sets `0x200001ee` (stream on/off).                                                                                                                              |

Not handled by the AT32 (falls to default): 69 `PC_SYNC_TEACH`, 70
`SYNC_WRITE_SERVOS`. The ESP32 handles those itself with raw bus frames.

### 3.2 The 10 ms tick and what it is for

- TMR6 is configured with prescaler 199 and period 5999 (`0x8006a04-0x8006a0a`).
  At the 120 MHz system clock that is 600 kHz / 6000 = **10 ms**, matching the
  vendor doc "AT32 pushes 6 servo positions every 10 ms".
- The TMR6 interrupt (`0x80035e0`) sets flag `0x200001e8` every tick and
  `0x200001e9` every 20 ticks (200 ms).
- Main loop on the 10 ms flag (`0x8005ac2-0x8005b44`):
  - if streaming (`0x1ee`): sync read positions and push a cmd 0x68 frame to
    the ESP32 (this is the teach-recording feed);
  - else if the LeRobot flag `0x1ed` is clear: run the coordinate-move
    interpolator (`0x8002d34`) and, when it has a new frame, sync write six
    records `{id, acc = move_acc, speed, pos}` (`0x8005658-0x8005692`).
- The first CMD 96 sets `0x1ed`, which **silences the interpolator and the
  200 ms status push for good**. After that, the only bus traffic is what the
  PC asks for. There is no firmware path that interpolates or rate-limits
  CMD 97.

So the "10 ms interpolation" the vendor SDK describes exists, but it serves
coordinate moves and action groups. **LeRobot writes bypass it entirely.**

### 3.3 What a CMD 96 read actually fetches

The sync read is `addr 0x38 (56), len 8` (`0x8005e34-0x8005e3c`). Register 56..63
on this servo family is present position, present speed, present load,
voltage, temperature. The wrapper (`0x8005e20`) parses position, speed and
load into 10-byte records; the CMD 96 handler copies only the position
halfword out of each record (`0x8004cb4-0x8004cc0`).

**Speed, load, voltage and temperature are on the bus in every frame already.
Only the last 8 bytes of the reply are missing.**

### 3.4 Servo register map

The AT32 and ESP32 code use: 5 ID, 6 baud, 21..25 PID block, 26/27 dead zones,
31 position offset, 33 mode, 34..36 protection, 40 torque enable, 41 acc, 42
goal position, 44 goal time/PWM, 46 goal speed, 48 max torque, 56 present
position, 58 present speed, 60 present load, 62 voltage, 63 temperature, 66
moving, 69 present current; commands 2 read, 3 write, 130 sync read, 131 sync
write; speed clamp ±3400; 4096 ticks per revolution (`HX_30HM_Def.h`,
`0x8005f08-0x8005f62`). This is the Feetech STS-series memory map. The HX-30HM
/ HX-65HM behave as STS-protocol servos, so Feetech's documented semantics
apply: register 21 P, 22 D, 23 I, 24..25 minimum startup force; goal speed 0
means "no limit"; acc 0 means "no ramp"; the position loop is a proportional
controller with a small default P (32 on STS3215).

No write to the EEPROM lock register (55) appears anywhere in the AT32 image.
On Feetech servos a locked EEPROM ignores writes to registers below 40.
Whether these HX servos lock is unknown; a CMD 60 readback after a CMD 59
write settles it.

## 4. What this changes in the earlier analyses

1. **`motion_speed` and `motion_acc` do nothing in LeRobot mode.** CMD 56 keeps
   one byte for the coordinate interpolator that CMD 97 never uses. This is the
   mechanism behind Finding 1 of the real-data report (joints track commands
   above 2000 ticks/s with `motion_speed=2000`). The docstring of
   `write_motion_params` in `src/lerobot/motors/nexarm/nexarm.py` ("Stored in
   AT32 as arm.lerobot_speed and applied each CMD 97") is wrong for this image;
   the comment on `CMD_SET_MOTION_PARAMS` is right.
2. **Every CMD 97 is a direct goal write with acc 0 and speed 0.** The servo
   moves at its own maximum toward the goal. The follower's dynamic response is
   therefore the servo's internal position loop and nothing else. The 2-frame
   command-to-state lag is: one frame of loop ordering (the action is sent
   after the observation), plus the servo's proportional approach with a low
   default P, plus about 5 to 10 ms of transport (estimate, §5).
3. **Gain authority exists.** CMD 59 writes the servo P/I/D block. That is the
   knob a "tracker" would be reshaping setpoints to imitate. Tune it first,
   with a step test, before any learned controller. Raise P in small steps and
   watch overshoot, hum and temperature; add D if it rings.
4. **Load telemetry is reachable without touching the AT32.** Two options:
   - normal (non-bridge) mode: the ESP32 forwards raw frames for IDs 1..6 to
     the bus (`func_ctrl_callback` else-branch) but drops the replies; relaying
     replies to USB is a small ESP32 patch, and then present load (60), current
     (69) and temperature (63) are readable per servo;
   - bridge mode: enable CMD 104 and relay cmd 0x68 frames in
     `bridge_at32_callback` to get a 100 Hz position stream without polling.
     Neither gives load inside the CMD 96 reply itself; that would need the AT32.
5. The "status byte 0x41" and "HX-30HM register table" in the pasted proposal
   were half right: register 41 is acceleration, and the 20 % overload default
   is the Feetech protection-torque default behind CMD 71/72.

## 5. Latency budget per 30 Hz frame (estimate)

| Step                                                     | Where                   | Time          |
| -------------------------------------------------------- | ----------------------- | ------------- |
| CMD 96 USB → ESP32 → AT32                                | CH340 + two 1 Mbps hops | 1 to 3 ms     |
| Sync read of 6 servos, 8 bytes each                      | AT32 USART2             | 1.5 to 2.5 ms |
| Reply AT32 → ESP32 → USB → Python poll loop (1 ms sleep) |                         | 2 to 4 ms     |
| Leader read, six sequential register reads               | leader ESP32            | 7 to 10 ms    |
| CMD 97 PC → bus (56-byte sync write)                     |                         | 1 to 2 ms     |

The host loop spends roughly 15 to 20 ms of every 33 ms frame in serial I/O,
which is why `lerobot-record` printed "loop is running slower" so often. The
leader read is the largest single item. The follower's `get_observation`
already logs its read time at debug level; one run with debug logging gives
the real numbers.

## 6. Recommended next steps

1. Fix the `write_motion_params` docstring and the config comments that say
   `motion_speed` caps moves. Keep the parameter (it still affects vendor
   coordinate moves) but say it has no effect on CMD 97.
2. Step test with `examples/nexarm/servo_step_test.py`, then read the PID block
   with CMD 60 on each servo, raise P on one joint, re-run the step test. This
   is the cheapest tracking improvement available and is the experiment that
   decides whether any controller work is worth doing.
3. Patch the ESP32 bridge to relay cmd 60/71/75 replies (so readback works in
   LeRobot mode) and cmd 0x68 stream frames. Keep the change in
   `bridge_at32_callback`; it is one `if`.
4. Add an optional load/current read path to `NexArmMotorsBus` once the relay
   exists, and log it during recording. Gripper load is the grasp detector the
   stack-bowls task is missing.
5. Replace the 2000 ticks/s speed cap in the sim plan with a per-joint
   proportional servo model (goal → position through a P loop with torque and
   speed saturation). That is what the hardware runs.

## Unresolved questions

1. Is the ESP32 Serial1 electrically on the servo bus? The firmware only makes
   sense if it is, but a scope or a raw ping with reply bytes visible on the
   ESP32 debug console is the proof.
2. Do the HX servos honour EEPROM writes to register 21 without unlocking
   register 55? Test with CMD 59 followed by CMD 60.
3. What are the factory P/I/D values on the six servos? Read them before
   changing anything, and record them in `docs/`.
4. CH340 USB latency on the record PC: measure the CMD 96 round trip from
   Python; the table above is an estimate.
