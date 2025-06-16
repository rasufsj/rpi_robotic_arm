import cv2 as cv
import numpy as np
import math

# === Camera calibration matrix (intrinsics) ===
# This matrix is obtained through camera calibration and defines focal length and principal point
camera_matrix = np.array([
    [897.47499574, 0.0, 324.82951238],
    [0.0, 897.13216739, 241.34067822],
    [0.0, 0.0, 1.0]
])

# === Distortion coefficients ===
# These values correct radial and tangential lens distortion
dist_coeffs = np.array([
    0.128032231, -1.50656891, -0.0156114801, 0.00223699925, 6.89414705
])

# === Helper function: Checks if a matrix is a valid rotation matrix ===
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    return np.linalg.norm(I - shouldBeIdentity) < 1e-6

# === Converts a rotation matrix to Euler angles (pitch, roll, yaw) ===
def rotationMatrixToEulerAngles(R):
    assert isRotationMatrix(R)
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
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

# === Simple 1D Kalman filter to smooth noisy position estimates ===
class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2, estimated_error=1.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_error = estimated_error
        self.posteri_estimate = 0.0
        self.posteri_error = 1.0

    def update(self, measurement):
        # Predict step
        priori_estimate = self.posteri_estimate
        priori_error = self.posteri_error + self.process_variance

        # Update step
        kalman_gain = priori_error / (priori_error + self.measurement_variance)
        self.posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)
        self.posteri_error = (1 - kalman_gain) * priori_error
        return self.posteri_estimate

# === ArUco detection and pose estimation class ===
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
        corners, ids, _ = self.detector.detectMarkers(frame)
        detections = []

        if ids is not None:
            # Estimate pose for each marker
            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, self.marker_length, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                position = tvecs[i][0]
                distance = np.linalg.norm(position)
                detections.append({
                    'id': ids[i][0],
                    'position': position,
                    'rotation': rvecs[i][0],
                    'distance': distance,
                    'corner': corners[i][0]
                })

            # Sort by distance (Z-axis)
            detections.sort(key=lambda d: d['position'][2])
            closest = detections[0]

            # Apply Kalman filtering
            if self.current_id != closest['id']:
                self.kf_x, self.kf_y, self.kf_z = SimpleKalmanFilter(), SimpleKalmanFilter(), SimpleKalmanFilter()
                self.current_id = closest['id']

            pos_filt = [
                self.kf_x.update(closest['position'][0]),
                self.kf_y.update(closest['position'][1]),
                self.kf_z.update(closest['position'][2])
            ]
            closest['position'] = pos_filt

            # Convert rotation vector to matrix
            rvec = closest['rotation']
            rvec_flipped = -rvec if self.use_flipped_pose else rvec
            R, _ = cv.Rodrigues(rvec_flipped)
            tvec_flipped = -np.array(pos_filt) if self.use_flipped_pose else np.array(pos_filt)

            # Calculate world translation and orientation
            closest['realworld_tvec'] = np.dot(R, tvec_flipped)
            closest['euler_angles_deg'] = np.degrees(rotationMatrixToEulerAngles(R))

            # Draw axes and markers
            cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, np.array([rvec_flipped]), np.array([pos_filt]), self.marker_length / 2)
            cv.aruco.drawDetectedMarkers(frame, corners, ids)
            corner = closest['corner']
            cv.putText(frame, f"ID {closest['id']}: {closest['distance']:.2f}m",
                       (int(corner[0][0]), int(corner[0][1]) - 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            return frame, [closest]
        else:
            self.current_id = None
            return frame, []
