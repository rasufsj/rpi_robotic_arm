import cv2
import numpy as np
import threading
import time

# ---------- Threaded Camera Class ----------
class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, 40)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.capture.read()
        self.running = True
        self.lock = threading.Lock()

        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()

    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def stop(self):
        self.running = False
        self.capture.release()

# ---------- Parâmetros da calibração ----------
chessboard_size = (8, 6)
square_size = 0.244

objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

# ---------- Inicialização da câmera ----------
cam = ThreadedCamera()
print("Pressione 's' para salvar frame válido. Pressione 'q' para terminar.")
saved = 0

cv2.namedWindow("Calibração", cv2.WINDOW_NORMAL)  # Janela redimensionável
cv2.resizeWindow("Calibração", 800, 600)

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.01)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, chessboard_size, corners, found)

        cv2.putText(display, f"Frames salvos: {saved}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Calibração", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and found:
            objpoints.append(objp)
            imgpoints.append(corners)
            saved += 1
            print(f"[INFO] Frame {saved} salvo.")
        elif key == ord('q'):
            break

        time.sleep(0.03)  # Controla taxa de atualização para não explodir a janela

finally:
    cam.stop()
    cv2.destroyAllWindows()

print(f"Total de imagens salvas: {saved}")
if saved < 5:
    print("⚠️ Salve ao menos 5 imagens com boa visão do tabuleiro.")
    exit()

# ---------- Calibração ----------
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("Matriz da câmera:\n", camera_matrix)
print("Coeficientes de distorção:\n", dist_coeffs)

np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)
print("Arquivos salvos: camera_matrix.npy, dist_coeffs.npy")
