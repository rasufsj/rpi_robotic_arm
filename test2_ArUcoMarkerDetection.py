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

        # Suavização + controle de ID atual
        self.buffer_posicao_cubo = None
        self.alpha = 0.7  # Fator de suavização
        self.id_atual = None

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
            
            # Ordena as detecções pela posição Z (menor primeiro)
            detections.sort(key=lambda det: det['position'][2])

            # Pega apenas a detecção mais próxima
            detection_mais_proxima = detections[0]
            id_detectado = detection_mais_proxima['id']
            posicao_atual = detection_mais_proxima['position']

            # Verifica se mudou o ID → reinicia o buffer se necessário
            if self.id_atual != id_detectado:
                self.buffer_posicao_cubo = posicao_atual
                self.id_atual = id_detectado
            else:
                # Suaviza posição
                ultima_pos = np.array(self.buffer_posicao_cubo)
                nova_pos = self.alpha * np.array(posicao_atual) + (1 - self.alpha) * ultima_pos
                self.buffer_posicao_cubo = nova_pos.tolist()

            # Atualiza no dicionário
            detection_mais_proxima['position'] = self.buffer_posicao_cubo

            # Desenha apenas a face mais próxima
            cv.drawFrameAxes(
                frame, 
                camera_matrix, 
                dist_coeffs, 
                np.array([detection_mais_proxima['rotation']]), 
                np.array([detection_mais_proxima['position']]), 
                self.marker_length / 2
            )

            cv.aruco.drawDetectedMarkers(frame, corners, ids)

            corner = detection_mais_proxima['corner']
            pos_texto = (int(corner[0][0]), int(corner[0][1]) - 30)
            cv.putText(frame, f"ID {detection_mais_proxima['id']}: {detection_mais_proxima['distance']:.2f}m", 
                      pos_texto,
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Retorna apenas 1 detecção em uma lista
            return frame, [detection_mais_proxima]
        
        else:
            # Nenhum marcador detectado → zera o buffer
            self.buffer_posicao_cubo = None
            self.id_atual = None
            return frame, []

def main():
    # Inicializa captura de vídeo
    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    if not cap.isOpened():
        print("Erro ao abrir câmera!")
        return
    
    # Cria detector
    detector = ArucoDetector(marker_length=0.04)  # Tamanho real que você mediu
    
    cv.namedWindow("Detecção ArUco", cv.WINDOW_NORMAL)
    
    # Controle de tempo para print
    last_print_time = time.time()
    print_interval = 8  # segundos
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame")
            break
        
        # Detecta marcadores
        processed_frame, detections = detector.detect(frame)
        
        # Print periódico
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            if detections:
                print("\n" + "="*50)
                print(f"Informações do cubo (face mais próxima) - {time.ctime(current_time)}")
                print("="*50)
                
                for detection in detections:
                    print(f"\nMarcador {detection['id']}:")
                    print(f"Posição (x,y,z): {detection['position']}")
                    print(f"Distância: {detection['distance']:.3f} metros")
                    print(f"Rotação (rx,ry,rz): {detection['rotation']}")
                
                print("\n" + "="*50 + "\n")
            else:
                print(f"\n[{time.ctime(current_time)}] Nenhum marcador detectado\n")
            
            last_print_time = current_time
        
        # Exibe a imagem
        cv.imshow("Detecção ArUco", processed_frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
