import cv2
import numpy as np
from fpdf import FPDF

# --- Configurações ---
TAMANHO_CUBO_MM = 60          # ALTERADO: 40x40mm (4cm)
MARGEM_BRANCA_MM = 5         # 10mm de margem branca ao redor
LINHA_CORTE_MM = 1            # Espessura da linha de corte tracejada
ESPACO_ENTRE_MM = 5           # Espaço entre marcadores na folha
DPI = 300                     # Resolução de impressão (300 DPI)

# --- Conversão mm → pixels ---
def mm_para_px(mm): 
    return int(mm * DPI / 25.4)

TAMANHO_MARCADOR_PX = mm_para_px(TAMANHO_CUBO_MM)
MARGEM_PX = mm_para_px(MARGEM_BRANCA_MM)
LINHA_CORTE_PX = mm_para_px(LINHA_CORTE_MM)
ESPACO_PX = mm_para_px(ESPACO_ENTRE_MM)

# --- Gera 6 IDs distintos para as 6 faces do cubo ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# IDs únicos para cada face do cubo
ids_cubo = [1, 2, 3, 4]  # ALTERADO: 6 IDs distintos
marcadores = []

for id_ in ids_cubo:
    # Gera o marcador ArUco (40mm)
    if hasattr(cv2.aruco, 'generateImageMarker'):
        marker = cv2.aruco.generateImageMarker(
            aruco_dict, id_, TAMANHO_MARCADOR_PX)
    else:
        marker = cv2.aruco.drawMarker(aruco_dict, id_, TAMANHO_MARCADOR_PX)

    # Adiciona margem branca (10mm) e linha de corte tracejada
    marker_com_margem = 255 * np.ones(
        (TAMANHO_MARCADOR_PX + 2*MARGEM_PX, TAMANHO_MARCADOR_PX + 2*MARGEM_PX),
        dtype=np.uint8
    )
    marker_com_margem[
        MARGEM_PX: MARGEM_PX + TAMANHO_MARCADOR_PX,
        MARGEM_PX: MARGEM_PX + TAMANHO_MARCADOR_PX
    ] = marker

    # Linha tracejada de corte (5px preto, 5px branco)
    for i in range(0, marker_com_margem.shape[0], 10):
        marker_com_margem[i:i+5, :LINHA_CORTE_PX] = 0    # Borda esquerda
        marker_com_margem[i:i+5, -LINHA_CORTE_PX:] = 0   # Borda direita
        marker_com_margem[:LINHA_CORTE_PX, i:i+5] = 0    # Borda superior
        marker_com_margem[-LINHA_CORTE_PX:, i:i+5] = 0   # Borda inferior

    marcadores.append(marker_com_margem)

# --- Organiza em folha A4 (2 linhas x 3 colunas) ---
LINHAS, COLUNAS = 6, 3
LARGURA_MARCADOR_TOTAL_PX = TAMANHO_MARCADOR_PX + 2*MARGEM_PX
ALTURA_MARCADOR_TOTAL_PX = TAMANHO_MARCADOR_PX + 2*MARGEM_PX

LARGURA_FOLHA_PX = COLUNAS * LARGURA_MARCADOR_TOTAL_PX + (COLUNAS-1)*ESPACO_PX
ALTURA_FOLHA_PX = LINHAS * ALTURA_MARCADOR_TOTAL_PX + (LINHAS-1)*ESPACO_PX
folha = 255 * np.ones((ALTURA_FOLHA_PX, LARGURA_FOLHA_PX), dtype=np.uint8)

for i, marcador in enumerate(marcadores):
    linha = i // COLUNAS
    coluna = i % COLUNAS
    x = coluna * (LARGURA_MARCADOR_TOTAL_PX + ESPACO_PX)
    y = linha * (ALTURA_MARCADOR_TOTAL_PX + ESPACO_PX)

    # Verifica se o marcador cabe na folha
    if y + marcador.shape[0] <= folha.shape[0] and x + marcador.shape[1] <= folha.shape[1]:
        folha[y:y+marcador.shape[0], x:x+marcador.shape[1]] = marcador
    else:
        print(f"Erro: Marcador {i+1} não cabe na folha!")

# --- Salva imagem e PDF ---
cv2.imwrite("cubo_aruco_4cm.png", folha)
print("[INFO] Imagem salva como 'cubo_aruco_4cm.png'")

pdf = FPDF(unit="mm", format="A4")
pdf.add_page()
pdf.image("cubo_aruco_4cm.png", x=10, y=10, w=190)  # Ajuste para margens
pdf.output("cubo_aruco_4cm.pdf")
print("[INFO] PDF salvo como 'cubo_aruco_4cm.pdf'")