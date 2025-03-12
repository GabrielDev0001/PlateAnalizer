import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

MODEL_PATH = "C:\\Users\\bh14\\Documents\\GitHub\\model\\10_02\\best.pt"
model = YOLO(MODEL_PATH)

def analyze_image(image_path, window_width=800, window_height=800, seguir=False, deixar_rastro=False):
    """
    Args:
        image_path (str): Caminho para a imagem a ser analisada.
        window_width (int): Largura desejada da janela de exibição.
        window_height (int): Altura desejada da janela de exibição.
        seguir (bool): Indica se a análise deve seguir os objetos detectados.
        deixar_rastro (bool): Indica se deve deixar rastros de objetos seguidos.
    """

    # Carregar a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return None, 0, ""

    # Obter as dimensões originais da imagem
    original_height, original_width = img.shape[:2]

    # Calcular o fator de escala para manter a proporção
    scale = min(window_width / original_width, window_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Redimensionar proporcionalmente
    img_resized = cv2.resize(img, (new_width, new_height))

    # Fazer a inferência
    if seguir:
        results = model.track(img, persist=True)
    else:
        results = model(img)

    # Histórico de rastreamento
    track_history = defaultdict(lambda: [])
    
    # Contador de labels e bactérias
    label_count = 0
    bacteria_count = defaultdict(int)

    # Processar resultados
    for result in results:
        boxes = result.boxes.xywh.cpu()  # (x_center, y_center, width, height)
        confidences = result.boxes.conf.cpu().tolist()
        class_indices = result.boxes.cls.cpu().tolist()  # Índices das classes preditas
        labels = result.names  # Dicionário de nomes das classes

        for box, conf, class_index in zip(boxes, confidences, class_indices):
            x_center, y_center, width, height = box
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            # Nome da bactéria
            bacteria_name = labels.get(int(class_index), "Desconhecido")
            if bacteria_name != "Desconhecido":
                bacteria_count[bacteria_name] += 1


            # Desenhar a caixa delimitadora
            cv2.rectangle(
                img, (x_min, y_min), (x_max, y_max), color=(161, 51, 0), thickness=3
            )

            # Adicionar o rótulo na imagem
            label_text = f"{bacteria_name}"
            cv2.putText(
                img, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
    
    # Criar uma string com nome e quantidade de cada bactéria
    bacteria_info = ", ".join([f"{name}: {count}" for name, count in bacteria_count.items()])

    # Redimensionar para exibição
    img_display = cv2.resize(img, (window_width, window_height), interpolation=cv2.INTER_LINEAR)

    # Retornar a imagem processada, a quantidade de labels e os nomes e quantidades das bactérias
    return img_display, bacteria_info

# Exemplo de uso da função
if __name__ == "__main__":
    processed_image, num_labels, bacteria_info = analyze_image(
        image_path="C:\\Users\\bh14\\Documents\\GitHub\\GerenciaViwer\\datasets\\petri\\test\\2973_20232.jpg",
        window_width=800,
        window_height=600,
        seguir=False,
        deixar_rastro=False
    )

    if processed_image is not None:
        print(f"Quantidade de labels detectados: {num_labels}")
        print(f"Bactérias detectadas: {bacteria_info}")

        # Exibir a imagem processada
        cv2.imshow("Resultados", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
