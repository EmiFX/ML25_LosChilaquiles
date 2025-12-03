import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import cv2
from ml25.P02_facial_expressions.othermodels.convnextTiny.networkconvnextTiny import Network
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
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")
    val_transforms, unnormalize = get_transforms("test_imgs", img_size=48, enhance=True)
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
    model_path = file_path / "othermodels" / "convnextTiny" / "models" / "best_model_tiny.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    modelo.load_model(str(model_path))
    modelo.eval()

    # cargarlo en CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo = modelo.to(device)
    
    paths = img_title_paths.values() if isinstance(img_title_paths, dict) else img_title_paths

    for path in paths:
        # Cargar la imagen
        try:
            # Convertir a Path si es string relativo
            if not os.path.isabs(path):
                im_file = (file_path / path).as_posix()
            else:
                im_file = path
                
            print(f"Procesando: {im_file}")
            original, transformed, denormalized = load_img(im_file)

            transformed = transformed.unsqueeze(0).to(device)

            # Inferencia
            with torch.no_grad():
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
        except Exception as e:
            print(f"Error procesando {path}: {str(e)}")
            continue
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Directorio de test images
    test_dir = file_path / "test_imgs"
    
    if not test_dir.exists():
        print(f"El directorio {test_dir} no existe")
        exit(1)
    
    # Get all image files
    img_paths = [str(p) for p in test_dir.iterdir() 
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not img_paths:
        print(f"No se encontraron imágenes en {test_dir}")
        exit(1)
        
    print(f"Encontradas {len(img_paths)} imágenes")
    predict(img_paths)