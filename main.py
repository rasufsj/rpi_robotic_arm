import cv2 as cv
import time
import serial
from ArUcoDetector import ArucoDetector

# === Dictionary mapping ArUco IDs to cube descriptions and destination shelves ===
id_to_cube_info = {
    1: {"cube_name": "Cube 1", "destination": "Shelf A"},
    2: {"cube_name": "Cube 2", "destination": "Shelf B"},
    3: {"cube_name": "Cube 3", "destination": "Shelf C"},
    4: {"cube_name": "Cube 4", "destination": "Shelf D"}
}

def main():
    # Start video capture from the default camera
    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    
    # Camera settings
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Reduces delay
    cap.set(cv.CAP_PROP_FPS, 40)
    
    if not cap.isOpened():
        print("❌ Camera not available.")
        return

    detector = ArucoDetector(marker_length=0.04, use_flipped_pose=True)

    # === Open serial connection to Arduino Nano via UART GPIO ===
    try:
        arduino = serial.Serial('/dev/serial0', 115200, timeout=1)
        time.sleep(2)  # Give time for Arduino to reboot
        print("\n🔌 Serial connection established.")
    except Exception as e:
        print(f"\n⚠️ Unable to open serial port: {e}")
        arduino = None

    cv.namedWindow("ArUco Detection", cv.WINDOW_NORMAL)
    last_print_time = time.time()  # Time control for printing
    print_interval = 5  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera.")
            break

        processed_frame, detections = detector.detect(frame)
        current_time = time.time()

        if current_time - last_print_time >= print_interval:
            if detections:
                print("\n" + "="*50)
                print(f"📡 Detected marker - {time.ctime(current_time)}")
                print("="*50)

                for d in detections:
                    marker_id = d['id']
                    cube = id_to_cube_info.get(marker_id, {})
                    name = cube.get("cube_name", "Unknown")
                    dest = cube.get("destination", "Shelf ?")

                    # Display detailed pose information
                    print(f"\n🟡 {name} → {dest} (Marker ID {marker_id})")
                    print(f"📍 Position: {d['position']}")
                    print(f"📏 Distance: {d['distance']:.3f} m")
                    print(f"🌍 RealWorld Tvec: {d['realworld_tvec']}")
                    print(f"🎛️ Euler Angles: Pitch={d['euler_angles_deg'][0]:.2f}°, "
                          f"Roll={d['euler_angles_deg'][1]:.2f}°, "
                          f"Yaw={d['euler_angles_deg'][2]:.2f}°")

                    # === Send command to Arduino Nano over serial ===
                    if arduino:
                        command = f"PICK_{marker_id}_TO_{dest[-1]}"
                        try:
                            arduino.write((command + "\n").encode())
                            print(f"🔁 Sent to Arduino: {command}")
                        except Exception as e:
                            print(f"⚠️ Serial write failed: {e}")

                print("\n" + "="*50 + "\n")
            else:
                print(f"[{time.ctime(current_time)}] No markers detected")

            last_print_time = current_time

        # Show camera feed
        cv.imshow("ArUco Detection", processed_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    if arduino:
        arduino.close()

if __name__ == "__main__":
    main()
