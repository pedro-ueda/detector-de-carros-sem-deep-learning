import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Configuração inicial da página web
st.set_page_config(page_title="Detector de Carros Clássico", layout="wide")

st.title("🚗 Simulador de Visão Computacional Baseado em Regras")
st.markdown(
    "Este sistema identifica veículos usando geometria e processamento de imagem clássico (sem Deep Learning)."
)

# 1. Painel Lateral de Controle (Calibração das Regras Heurísticas)
st.sidebar.header("🔧 Parâmetros de Calibração")

min_area = st.sidebar.slider(
    "Área Mínima do Contorno (px)",
    min_value=500,
    max_value=20000,
    value=3000,
    step=500,
)
max_area = st.sidebar.slider(
    "Área Máxima do Contorno (px)",
    min_value=20000,
    max_value=1000000,
    value=500000,
    step=10000,
)

st.sidebar.subheader("Proporção do Veículo (Aspect Ratio)")
min_ratio = st.sidebar.slider(
    "Largura / Altura Mínima", min_value=0.5, max_value=2.0, value=1.2, step=0.1
)
max_ratio = st.sidebar.slider(
    "Largura / Altura Máxima", min_value=2.0, max_value=5.0, value=3.5, step=0.1
)

# 2. Upload da Imagem
upload = st.file_uploader(
    "Escolha uma imagem de trânsito ou estacionamento...",
    type=["jpg", "jpeg", "png"],
)

if upload is not None:
    # Converter o upload do Streamlit para o formato que o OpenCV entende
    image_pil = Image.open(upload)
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    img_output = img_bgr.copy()

    # 3. Pipeline de Processamento Clássico
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Operação morfológica para juntar bordas próximas do chassi
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Encontrar os contornos
    contornos, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    carros_detectados = 0

    # 4. Aplicação das Regras com base nos Sliders
    for contorno in contornos:
        area = cv2.contourArea(contorno)

        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = float(w) / h

            if min_ratio <= aspect_ratio <= max_ratio:
                carros_detectados += 1
                # Desenha na imagem (padrão BGR do OpenCV)
                cv2.rectangle(img_output, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(
                    img_output,
                    f"Carro {carros_detectados}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    # Converter de volta para RGB para exibição correta no Streamlit
    img_output_rgb = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)

    # 5. Exibição dos Resultados na Tela
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Imagem Original")
        st.image(image_pil, use_container_width=True)

    with col2:
        st.subheader("🔍 Resultado da Detecção")
        st.image(img_output_rgb, use_container_width=True)

    # Veredito em formato de Banner
    st.markdown("---")
    if carros_detectados > 0:
        st.success(
            f"**Análise Concluída:** {carros_detectados} padrão(ões) de veículo identificado(s) com sucesso!"
        )
    else:
        st.warning(
            "**Análise Concluída:** Nenhum veículo detectado com os parâmetros atuais. Tente ajustar os controles laterais."
        )