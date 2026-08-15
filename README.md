# Exoskeleton 

## Overview

Exoskeleton is a powered lower-limb exoskeleton with six actuated joints — hip flexion, hip abduction, and knee flexion per leg. Each joint is driven by a CubeMars AK-series actuator (AK70-10 / AK80-64) commanded over CAN by a Teensy 4.1 running through a custom dual-CAN power distribution board, which supplies six independent motor power outputs. The frame — waist, hip abductors, hip-to-thigh, thigh-to-knee, and knee-to-shank sections — is designed in SolidWorks; the electronics are designed in KiCad.

## Features

- Six actuated joints (hip flexion, hip abduction, knee) via CubeMars AK70-10 / AK80-64 actuators
- Custom power distribution board: XT90 main input, 6× XT30 motor power outputs (one per actuator), 50A MIDI fuse, dump resistor, thermal switch input, and battery voltage sense
- Dual CAN bus (CAN A / CAN B, SN65HVD230 transceivers) for actuator communication
- Teensy 4.1 as the main controller
- Modular frame: waist (left/middle/right), hip abductors, hip-to-waist and hip-to-thigh frames, thigh-to-knee frames, knee-to-shank frames, and hip/knee step-up and step-down joint hardware

## Repository layout

- `SolidWorks Assembly/July 2026/July 2026 Exo Assembly/` — SolidWorks parts and the top-level assembly (`Pitt Exo - Final July 2026 Assembly.SLDASM`)
- `pcb/` — KiCad project for the power distribution / control board (schematic, PCB layout, fabrication outputs)
- `pcb/production/` — BOM, designators, placement data, IPC netlist, and Gerber/drill ZIP for fabrication
- `Digikey Order`, `Exoskeleton-JLCPCB Assembly Order (1).xls` — component and assembly ordering records

## Bill of materials

The authoritative electrical BOM is [`pcb/production/bom.csv`](pcb/production/bom.csv). In summary, the power/control board uses:

- 1 Teensy 4.1 controller
- 2 SN65HVD230 CAN transceivers (CAN A / CAN B)
- 1 XT90 power input and 6 XT30 motor power connectors
- 1 50A MIDI fuse, 1 IRFB4110 MOSFET, dump resistor output, thermal switch input, and battery sense header
- Supporting passives (bulk 680 µF caps, TVS/flyback diodes, 120 Ω CAN bus termination)

The mechanical assembly (SolidWorks) is built from 22 parts, including the waist sections, left/right hip abductors, hip-to-waist and hip-to-thigh frames, thigh-to-knee and knee-to-shank frames, hip/knee step-up and step-down joint hardware, and the AK70-10 / AK80-64 motor body and center mounts.

## Pictures

<img width="726" height="845" alt="image" src="https://github.com/user-attachments/assets/03b4c0f5-4bc2-48e9-bba4-621ab739fc5e" />

<img width="1676" height="930" alt="Exoskeleton" src="https://github.com/user-attachments/assets/556e070b-1326-449b-a57e-b67a87f2d5f0" />

<img width="629" height="582" alt="image" src="https://github.com/user-attachments/assets/017c70a7-cbbf-40d1-b131-a537bfd3d297" />
