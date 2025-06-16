import cv2 as cv

# Tenta abrir a webcam padrão
cap = cv.VideoCapture(0)

# ✅ Configurações da câmera
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Reduz o atraso

if not cap.isOpened():
    print("❌ Erro: Não foi possível acessar a webcam.")
    exit()

print("✅ Webcam detectada. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Erro ao capturar frame da webcam.")
        break

    # Exibe a imagem capturada
    cv.imshow('Webcam - Genius 1000x HD V2', frame)

    # Sai com a tecla 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
