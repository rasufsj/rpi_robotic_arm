# 🤖 ArUco Cube Detection & Robotic Arm Control

This project uses computer vision on a **Raspberry Pi 4** with a connected **Logitech C270 Webcam** to detect cubes marked with **ArUco codes**. Each cube has a destination shelf assigned based on its marker ID.

Once detected, the Raspberry Pi calculates the 3D position and sends a serial command via **UART USB** to an **Arduino Nano**, which controls a **cartesian robotic arm** to move the cube.

---

## 🎯 Objectives

- Detect ArUco markers using OpenCV
- Estimate 3D position and orientation of each cube
- Assign a shelf destination based on marker ID
- Send commands to Arduino Nano via UART (USB)
- Execute pick-and-place using a robotic arm

---

## 📂 Project Structure

```
project/
├── ArUcoDetector.py       # Vision and pose estimation with Kalman filtering
├── main.py                # Main application loop (camera + serial communication)
├── sketch_codigo_projeto  # Complete Arduino Nano Code
├── calib/                 # Camera calibration files
└── README.md
```

---

## 🛠️ Requirements

### Raspberry Pi:

- Raspberry Pi 4 Model B (or similar)
- Python 3.9+
- OpenCV (with ArUco support)
- Camera (USB, RealSense, etc.)
- Serial UART enabled

### Python Dependencies:

```bash
pip install opencv-python numpy
```

---

## 🚀 How It Works

1. `main.py` captures frames from the camera.
2. `ArUcoDetector.py` detects markers, filters positions using a Kalman filter, and estimates the 3D pose.
3. Based on the marker ID, it chooses a target shelf (A, B, C, D).
4. Sends a command to the Arduino Nano via `/dev/serial0`, like:
   ```
   PICK_2_TO_B
   ```
5. Arduino receives the command and moves the robotic arm accordingly.

---

## 🔁 Communication Protocol

- Interface: UART over USB (e.g., `/dev/ttyUSB0` on Linux or `COM3` on Windows)
- Message format:
  ```
  PICK_<marker_id>_TO_<shelf_letter>
  ```

Example:

```text
PICK_3_TO_C  →  Pick cube 3 and place it on Shelf C
```

---

## 📦 ArUco Marker Mapping

| Marker ID | Cube Name | Shelf Destination |
| --------- | --------- | ----------------- |
| 1         | Cube 1    | Shelf A           |
| 2         | Cube 2    | Shelf B           |
| 3         | Cube 3    | Shelf C           |
| 4         | Cube 4    | Shelf D           |

---

## 🔧 Arduino Nano

- Connect via UART (USB) with voltage converter
- Listens for `PICK_<id>_TO_<shelf>` and executes pick-and-place motion

---

## 🗄️ Screenshots

> *You can add a screenshot here from the camera window detecting markers and drawing axes*

---

## 📌 Notes

- You can expand this to support multiple markers per cube (e.g. take average position)
- Add YOLO detection for classifying objects if needed in the future
- The code is modular and ready for integration with a physical robotic arm
