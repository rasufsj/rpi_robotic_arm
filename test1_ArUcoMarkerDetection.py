import cv2 as cv
import numpy as np
import math
import time

# Carrega os parâmetros da câmera
try:
    camera_matrix = np.array([
        [897.47499574, 0.0, 324.82951238],
        [0.0, 897.13216739, 241.34067822],
        [0.0, 0.0, 1.0]
    ])
    
    dist_coeffs = np.array([1.28032231e-01, -1.50656891e+00, -1.56114801e-02, 2.23699925e-03, 6.89414705e+00])
    
    print("Matriz da câmera carregada com sucesso!")
    print(camera_matrix)
    print("\nCoeficientes de distorção carregados com sucesso!")
    print(dist_coeffs)

except Exception as e:
    print("Erro ao carregar parâmetros da câmera:", e)
    exit(1)

class ArucoDetector:
    def __init__(self, marker_length=0.04):  # Tamanho do marcador em metros
        self.marker_length = marker_length
        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        self.parameters = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.aruco_dict, self.parameters)

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
                    'distance': distance  # Adiciona a distância total
                })
                
                cv.drawFrameAxes(
                    frame, 
                    camera_matrix, 
                    dist_coeffs, 
                    rvecs[i], 
                    tvecs[i], 
                    self.marker_length/2
                )
                
                # Exibe a distância no frame
                cv.putText(frame, f"ID {ids[i][0]}: {distance:.2f}m", 
                          (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 30),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            frame = cv.aruco.drawDetectedMarkers(frame, corners, ids)
        
        return frame, detections

def main():
    # Inicializa captura de vídeo
    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    if not cap.isOpened():
        print("Erro ao abrir câmera!")
        return
    
    # # Configurações da câmera
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv.CAP_PROP_FPS, 30)
    
    # Cria detector
    detector = ArucoDetector(marker_length=0.04)  # Ajuste o tamanho real do seu marcador
    
    cv.namedWindow("Detecção ArUco", cv.WINDOW_NORMAL)
    
    # Variável para controle do tempo
    last_print_time = time.time()
    print_interval = 5  # Intervalo em segundos
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame")
            break
        
        # Detecta marcadores
        processed_frame, detections = detector.detect(frame)
        
        # Verifica se passaram 5 segundos desde o último print
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            if detections:
                print("\n" + "="*50)
                print(f"Informações dos marcadores - {time.ctime(current_time)}")
                print("="*50)
                
                for detection in detections:
                    print(f"\nMarcador {detection['id']}:")
                    print(f"Posição (x,y,z): {detection['position']}")
                    print(f"Distância: {detection['distance']:.3f} metros")
                    print(f"Rotação (rx,ry,rz): {detection['rotation']}")
                
                print("\n" + "="*50 + "\n")
            else:
                print(f"\n[{time.ctime(current_time)}] Nenhum marcador detectado\n")
            
            last_print_time = current_time  # Atualiza o tempo do último print
        
        cv.imshow("Detecção ArUco", processed_frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()