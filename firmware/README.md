# NexArm firmware bundle

This directory contains the Hiwonder NexArm firmware artifacts, organized by
target and purpose. The sketches carry the LeRobot fixes listed under
[Changes from the vendor firmware](#changes-from-the-vendor-firmware); the
prebuilt images under `binaries/` are the unchanged vendor images.

## Layout

```text
firmware/
├── binaries/                    Prebuilt, merged ESP32 flash images
├── config/                      Vendor configuration data
├── docs/                        Protocol and firmware notes
├── esp32/
│   ├── follower/Nex_Arm/        Full follower/control-board Arduino sketch
│   └── leader/Nex_Arm/          Leader/synchronizer Arduino sketch
├── examples/
│   ├── k230_mcp/                K230 MCP demonstration
│   └── ros2/                    Vendor ROS 2 node and serial SDK
└── reference/
    ├── at32/                    Vendor workspace metadata for missing AT32 sources
    └── leader_variants/         Alternate leader sketches supplied as text files
```

## Firmware targets

### Follower ESP32

Open `esp32/follower/Nex_Arm/Nex_Arm.ino` as the Arduino sketch. It contains the
full control-board firmware and communicates with an AT32F421 over `Serial1`.
The embedded AT32 application image is stored in `at32_firmware.h` and may be
installed automatically by the ESP32 firmware at startup.

### Leader ESP32

Open `esp32/leader/Nex_Arm/Nex_Arm.ino` as the Arduino sketch. It communicates
directly with the six HX-30HM servos and supports ESP-NOW synchronization and
the LeRobot serial commands used to read joint positions.

### Prebuilt images

The files under `binaries/` are merged ESP32 images intended to be written from
address `0x0000`:

- `NexArm_follower_V1.0_0x0000.bin`
- `NexArm_leader_V1.0_0x0000.bin`

Keep these files unchanged as recovery artifacts. Confirm the controller role,
ESP32 board revision, flash size, and Hiwonder flashing procedure before using
them.

## Changes from the vendor firmware

- **Follower:** entering LeRobot mode (CMD 68) turns off ESP-NOW sync and the
  radio, and drops BLE commands, so the leader's ESP-NOW broadcast can no longer
  override the PC's CMD 97 writes. CMD 68 with 0 leaves bridge mode without a
  reboot. Firmware version 1.0.2.
- **Leader:** the USB parser keeps its state between reads, so commands split
  across two reads are no longer dropped; replies carry a checksum over their
  data; the `[POS]`/`[WARN]` debug text is off in LeRobot mode.

## Build and flash

Arduino-ESP32 2.0.12 (ESP-IDF 4.4.5, as recorded in the vendor images), board
`esp32:esp32:esp32`, partition scheme `huge_app` (the vendor images use the same
layout), and the U8g2, MultiButton and PS3 Controller Host libraries:

```bash
arduino-cli core install esp32:esp32@2.0.12 \
    --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli lib install U8g2 MultiButton "PS3 Controller Host"

FQBN=esp32:esp32:esp32:PartitionScheme=huge_app
arduino-cli compile -b "$FQBN" --build-path build/follower firmware/esp32/follower/Nex_Arm
arduino-cli compile -b "$FQBN" --build-path build/leader firmware/esp32/leader/Nex_Arm
```

Check which port is which before flashing: the leader prints `[POS] ...` lines
about once a second, the follower answers CMD 1 with its version. Back up the
board first, then upload; the upload writes the bootloader, partition table and
app, and keeps the saved settings (NVS) and action groups (LittleFS):

```bash
esptool.py --chip esp32 -p /dev/ttyUSB1 -b 921600 read_flash 0 0x400000 follower_backup.bin
arduino-cli upload -p /dev/ttyUSB1 -b "$FQBN" --input-dir build/follower firmware/esp32/follower/Nex_Arm
arduino-cli upload -p /dev/ttyUSB0 -b "$FQBN" --input-dir build/leader firmware/esp32/leader/Nex_Arm
```

To restore a backup, write it back from address 0:
`esptool.py --chip esp32 -p <port> -b 921600 write_flash 0 <backup>.bin`.

## Build limitations

Library versions are not pinned (the build above used U8g2 2.36.19,
MultiButton 1.3.0 and PS3 Controller Host 1.1.0). The AT32 bootloader and
application source trees referenced by `docs/AT32_OTA.md` are absent; only the
generated AT32 image embedded in the follower sketch is present.
