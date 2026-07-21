# NexArm firmware bundle

This directory contains the Hiwonder NexArm firmware artifacts, organized by
target and purpose. The files were reorganized without changing firmware logic
or binary contents.

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

## Build limitations

This bundle does not include a reproducible build configuration or pinned
Arduino library dependencies. The AT32 bootloader and application source trees
referenced by `docs/AT32_OTA.md` are also absent; only the generated AT32 image
embedded in the follower sketch is present.

The downloaded binaries contain build metadata for Arduino-ESP32 2.0.12 and
ESP-IDF 4.4.5. That metadata is useful for reconstructing the environment, but
it is not sufficient to guarantee a byte-identical rebuild.
