import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import cv2
from ml25.P02_facial_expressions.network import Network
import torch
from ml25.P02_facial_expressions.utils import (
    to_numpy,
    get_transforms,
    add_img_text,
)
from ml25.P02_facial_expressions.dataset import EMOTIONS_MAP
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()


def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path)
    val_transforms, unnormalize = get_transforms("test", img_size=48)
    tensor_img = val_transforms(img)
    denormalized = unnormalize(tensor_img)
    return img, tensor_img, denormalized


def predict(img_title_paths):
    """
    Hace la inferencia de las imagenes
    args:
    - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    """
    # Cargar el modelo
    modelo = Network(48, 7)
    modelo.load_model("best_model.pth")
    modelo.eval()

    # cargarlo en CUDA
    device = next(modelo.parameters()).device
    paths = img_title_paths.values() if isinstance(img_title_paths, dict) else img_title_paths

    for path in paths:
        # Cargar la imagen
        # np.ndarray, torch.Tensor
        im_file = (file_path / path).as_posix()
        original, transformed, denormalized = load_img(im_file)

        transformed = transformed.unsqueeze(0).to(device)

        # Inferencia
        logits, proba = modelo.predict(transformed)
        pred = torch.argmax(proba, -1).item()
        pred_label = EMOTIONS_MAP[pred]
        confidence = proba[0, pred].item() * 100

        # Original / transformada
        h, w = original.shape[:2]
        resize_value = 300
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        img = add_img_text(img, f"Pred: {pred_label} ({confidence:.1f}%)")

        # Mostrar la imagen
        denormalized = to_numpy(denormalized)
        denormalized = cv2.resize(denormalized, (resize_value, resize_value))
        cv2.imshow("Predicción - original", img)
        cv2.imshow("Predicción - transformed", denormalized)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Directorio de test images
    test_dir = file_path / "test_imgs"
    
    # Get all image files
    img_paths = [str(p) for p in test_dir.iterdir() 
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    predict(img_paths)