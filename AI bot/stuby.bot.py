from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


def get_class(model_path, labels_path, image_path):
    np.set_printoptions(suppress=True)

    # Carrega o modelo
    model = load_model(model_path, compile=False)

    # Carrega os nomes das classes
    class_names = open(labels_path, "r", encoding="utf-8").readlines()

    # Prepara a imagem
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(
        image,
        (224, 224),
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image)

    normalized_image_array = (
        image_array.astype(np.float32) / 127.5
    ) - 1

    data[0] = normalized_image_array

    # Faz a previsão
    prediction = model.predict(data, verbose=0)

    index = np.argmax(prediction)
    class_name = class_names[index].strip()[2:]
    confidence_score = prediction[0][index]

    return class_name, confidence_score


def gerar_resumo(classe, confianca):
    """Transforma a classe em uma descrição curta."""

    resumos = {
        "gato": "A imagem mostra um gato.",
        "cachorro": "A imagem mostra um cachorro.",
        "pessoa": "A imagem contém uma pessoa.",
        "carro": "A imagem apresenta um carro.",
        "passaro": "A imagem mostra um pássaro.",
        "comida": "A imagem contém algum tipo de alimento."
    }

    resumo = resumos.get(
        classe.lower(),
        f"A imagem parece conter: {classe}."
    )

    return (
        f"{resumo} "
        f"O modelo tem aproximadamente "
        f"{confianca * 100:.1f}% de confiança."
    )


# Exemplo de uso
modelo = "keras_model.h5"
labels = "labels.txt"
imagem = "foto.jpg"

classe, confianca = get_class(
    modelo,
    labels,
    imagem
)

print("Classe detectada:", classe)
print("Confiança:", f"{confianca * 100:.1f}%")

resumo = gerar_resumo(classe, confianca)

print("\nResumo:")
print(resumo)