import cv2 as cv
import numpy as np
import math
import time

# Load camera calibration parameters
try:
    camera_matrix = np.array([
        [897.47499574, 0.0, 324.82951238],
        [0.0, 897.13216739, 241.34067822],
        [0.0, 0.0, 1.0]
    ])
    
    dist_coeffs = np.array([1.28032231e-01, -1.50656891e+00, -1.56114801e-02, 2.23699925e-03, 6.89414705e+00])
    
# try:
#     camera_matrix = np.array([
#         [912.85582026, 0.0, 369.57386882],
#         [0.0, 911.704299, 303.69738201],
#         [0.0, 0.0, 1.0]
#     ])
    
#     dist_coeffs = np.array([0.01886467, -0.59032158, -0.0017988, 0.00805257, 0.97352338])
    
    print("Camera matrix loaded successfully!")
    print(camera_matrix)
    print("\nDistortion coefficients loaded successfully!")
    print(dist_coeffs)

except Exception as e:
    print("Error loading camera parameters:", e)
    exit(1)

# Mapping ArUco IDs to cube names and destinations
id_to_cube_info = {
    1: {"cube_name": "Cube 1", "destination": "Shelf A"},
    2: {"cube_name": "Cube 2", "destination": "Shelf B"},
    3: {"cube_name": "Cube 3", "destination": "Shelf C"},
    4: {"cube_name": "Cube 4", "destination": "Shelf D"}
}

# Rotation utilities
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    
    return np.array([x, y, z])

# Simple Kalman Filter 1D
class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2, estimated_error=1.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_error = estimated_error
        self.posteri_estimate = 0.0
        self.posteri_error = 1.0

    def update(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error = self.posteri_error + self.process_variance

        kalman_gain = priori_error / (priori_error + self.measurement_variance)
        self.posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)
        self.posteri_error = (1 - kalman_gain) * priori_error

        return self.posteri_estimate

# ArUco detector class
class ArucoDetector:
    def __init__(self, marker_length=0.04, use_flipped_pose=True):
        self.marker_length = marker_length
        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        self.parameters = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.kf_x = SimpleKalmanFilter()
        self.kf_y = SimpleKalmanFilter()
        self.kf_z = SimpleKalmanFilter()

        self.current_id = None
        self.use_flipped_pose = use_flipped_pose

    def detect(self, frame):
        corners, ids, rejected = self.detector.detectMarkers(frame)
        detections = []

        if ids is not None:
            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
                corners, 
                self.marker_length, 
                camera_matrix, 
                dist_coeffs
            )
            
            for i in range(len(ids)):
                position = tvecs[i][0]
                distance = math.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
                
                detections.append({
                    'id': ids[i][0],
                    'position': position,
                    'rotation': rvecs[i][0],
                    'distance': distance,
                    'corner': corners[i][0]
                })
            
            detections.sort(key=lambda det: det['position'][2])

            closest_detection = detections[0]
            detected_id = closest_detection['id']
            current_position = closest_detection['position']

            if self.current_id != detected_id:
                self.kf_x = SimpleKalmanFilter()
                self.kf_y = SimpleKalmanFilter()
                self.kf_z = SimpleKalmanFilter()
                self.current_id = detected_id

            x_filtered = self.kf_x.update(current_position[0])
            y_filtered = self.kf_y.update(current_position[1])
            z_filtered = self.kf_z.update(current_position[2])

            filtered_position = [x_filtered, y_filtered, z_filtered]
            closest_detection['position'] = filtered_position

            rvec = closest_detection['rotation']
            if self.use_flipped_pose:
                rvec_flipped = rvec * -1
            else:
                rvec_flipped = rvec

            rotation_matrix, _ = cv.Rodrigues(rvec_flipped)

            if self.use_flipped_pose:
                tvec_flipped = np.array(filtered_position) * -1
                realworld_tvec = np.dot(rotation_matrix, tvec_flipped)
            else:
                realworld_tvec = filtered_position

            euler_angles = rotationMatrixToEulerAngles(rotation_matrix)
            euler_angles_deg = np.degrees(euler_angles)

            closest_detection['realworld_tvec'] = realworld_tvec
            closest_detection['euler_angles_deg'] = euler_angles_deg

            cv.drawFrameAxes(
                frame, 
                camera_matrix, 
                dist_coeffs, 
                np.array([rvec_flipped]), 
                np.array([filtered_position]), 
                self.marker_length / 2
            )

            cv.aruco.drawDetectedMarkers(frame, corners, ids)

            corner = closest_detection['corner']
            text_position = (int(corner[0][0]), int(corner[0][1]) - 30)
            cv.putText(frame, f"ID {closest_detection['id']}: {closest_detection['distance']:.2f}m", 
                      text_position,
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            return frame, [closest_detection]
        
        else:
            self.current_id = None
            return frame, []

# Main application loop
def main():
    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    if not cap.isOpened():
        print("Error opening camera!")
        return
    
    detector = ArucoDetector(marker_length=0.04, use_flipped_pose=True)
    
    cv.namedWindow("ArUco Detection", cv.WINDOW_NORMAL)
    
    last_print_time = time.time()
    print_interval = 5  # seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame")
            break
        
        processed_frame, detections = detector.detect(frame)
        
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            if detections:
                print("\n" + "="*50)
                print(f"Marker info (closest face) - {time.ctime(current_time)}")
                print("="*50)
                
                for detection in detections:
                    marker_id = detection['id']
                    
                    cube_name = id_to_cube_info.get(marker_id, {}).get("cube_name", "Unknown Cube")
                    destination = id_to_cube_info.get(marker_id, {}).get("destination", "Unknown Shelf")
                    
                    print(f"\n🟡 {cube_name} → {destination} (Marker ID {marker_id})")
                    print(f"📍 Position (x, y, z):")
                    print(f"    X = {detection['position'][0]:.4f} m")
                    print(f"    Y = {detection['position'][1]:.4f} m")
                    print(f"    Z = {detection['position'][2]:.4f} m")
                    print(f"📏 Distance: {detection['distance']:.3f} m")
                    
                    print(f"🌍 Real World Translation Vector (X, Y, Z):")
                    print(f"    X = {detection['realworld_tvec'][0]:.4f}")
                    print(f"    Y = {detection['realworld_tvec'][1]:.4f}")
                    print(f"    Z = {detection['realworld_tvec'][2]:.4f}")
                    
                    print(f"🎛️ Euler Angles (deg):")
                    print(f"    Pitch = {detection['euler_angles_deg'][0]:.2f}°")
                    print(f"    Roll  = {detection['euler_angles_deg'][1]:.2f}°")
                    print(f"    Yaw   = {detection['euler_angles_deg'][2]:.2f}°")
                
                print("\n" + "="*50 + "\n")
            else:
                print(f"\n[{time.ctime(current_time)}] No markers detected\n")
            
            last_print_time = current_time
        
        cv.imshow("ArUco Detection", processed_frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
