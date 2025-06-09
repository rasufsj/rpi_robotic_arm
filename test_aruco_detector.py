import numpy as np
import cv2 as cv

ARUCO_DICT = {
    "DICT_4X4_50": cv.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv.aruco.DICT_4X4_1000
}
types = ['DICT_4X4_50']
arucoParams = cv.aruco.DetectorParameters()

def aruco_display(frame, types, arucoParams):
    all_corners, all_ids, typed = [], [], []
    for type_name in types:
        arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[type_name])
        corners, ids, rejected = cv.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        if ids is not None and len(corners) > 0:
            ids = ids.flatten()
            for (markerCorner, markerID) in zip(corners, ids):
                corners_reshaped = markerCorner.reshape((4, 2))
                all_corners.append(corners_reshaped)
                all_ids.append(markerID)
                typed.append(type_name)
    for i, corner in enumerate(all_corners):
        marker_id = all_ids[i]
        frame = draw_aruco(corner, marker_id, frame)
    return frame

def draw_aruco(corner, id, frame):
    (topLeft, topRight, bottomRight, bottomLeft) = corner
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))

    cv.line(frame, topLeft, topRight, (0, 255, 0), 2)
    cv.line(frame, topRight, bottomRight, (0, 255, 0), 2)
    cv.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
    cv.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
    cv.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

    cv.putText(frame, str(id), (topLeft[0], topLeft[1] - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Tenta diferentes índices e backends
for i in range(0, 5):
    cap = cv.VideoCapture(i, cv.CAP_V4L2)  # Tenta índice i com driver V4L2
    if cap.isOpened():
        print(f"Câmera encontrada no índice {i}")
        break
else:
    print("Erro: Nenhuma câmera detectada!")
    exit(1)

cv.namedWindow("Video", cv.WINDOW_NORMAL)

while True:
    success, frame = cap.read()
    if not success:
        print("Erro ao capturar frame.")
        break

    detected_markers = aruco_display(frame, types, arucoParams)
    cv.imshow("Video", detected_markers)
    
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()