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


def detect_face(img):
    """
    Intenta detectar una cara en la imagen con parámetros más estrictos.
    - Si encuentra cara, regresa el recorte (ROI) de la primera cara.
    - Si no encuentra, regresa None.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detección con parámetros más estrictos para reducir falsos positivos
    # scaleFactor: 1.2 (más estricto que 1.1)
    # minNeighbors: 8 (más alto = más estricto, reduce falsos positivos)
    # minSize: (60, 60) (cara mínima más grande)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2,      # Búsqueda más conservadora
        minNeighbors=8,        # Requiere más vecinos para confirmar
        minSize=(60, 60),      # Tamaño mínimo más grande
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        valid_faces = []
        for (x, y, w, h) in faces:
            aspect_ratio = w / h
            if 0.75 <= aspect_ratio <= 1.3:
                valid_faces.append((x, y, w, h))
        
        if len(valid_faces) > 0:
            x, y, w, h = valid_faces[0]
            return img[y:y+h, x:x+w]
    else:
        print("No se detecto la cara en la imagen.")
        return None


def load_img(img_or_path):
    """
    Carga y prepara una imagen para el modelo
    args:
    - img_or_path: path de la imagen o imagen BGR ya cargada
    returns:
    - img: imagen original
    - tensor_img: tensor transformado
    - denormalized: imagen denormalizada para visualización
    """
    if isinstance(img_or_path, str):
        assert os.path.isfile(img_or_path), f"El archivo {img_or_path} no existe"
        img = cv2.imread(img_or_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {img_or_path}")
    else:
        img = img_or_path
    
    val_transforms, unnormalize = get_transforms("test_imgs", img_size=48, enhance=True)
    tensor_img = val_transforms(img)
    denormalized = unnormalize(tensor_img)
    return img, tensor_img, denormalized


def predict(img_title_paths):
    """
    Hace la inferencia de las imagenes detectando primero las caras
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
        try:
            # Convertir a Path si es string relativo
            if not os.path.isabs(path):
                im_file = (file_path / path).as_posix()
            else:
                im_file = path
                
            print(f"\nProcesando: {im_file}")
            
            # Cargar imagen original
            original = cv2.imread(im_file)
            if original is None:
                raise ValueError(f"No se pudo cargar la imagen: {im_file}")
            
            # Detectar y recortar cara
            face_img = detect_face(original)
            
            # Si no se detectó cara, saltar esta imagen
            if face_img is None:
                print(f"Saltando imagen (sin caras detectadas)\n")
                continue
            
            # Preparar la cara para el modelo
            _, transformed, denormalized = load_img(face_img)
            transformed = transformed.unsqueeze(0).to(device)

            # Inferencia
            with torch.no_grad():
                logits, proba = modelo.predict(transformed)
            pred = torch.argmax(proba, -1).item()
            pred_label = EMOTIONS_MAP[pred]
            confidence = proba[0, pred].item() * 100
            
            # Mostrar todas las probabilidades (accuracy por emoción)
            print(f"Predicción: {pred_label} ({confidence:.1f}%)")
            print(f"Accuracy por emoción:")
            for emo_idx, emo_name in EMOTIONS_MAP.items():
                acc = proba[0, emo_idx].item() * 100
                print(f"      {emo_name}: {acc:.2f}%")

            # Preparar visualización de la cara recortada
            h, w = face_img.shape[:2]
            resize_value = 300
            face_display = cv2.resize(face_img, (w * resize_value // h, resize_value))
            face_display = add_img_text(face_display, f"Pred: {pred_label} ({confidence:.1f}%)")

            # Mostrar la imagen transformada
            denormalized_np = to_numpy(denormalized)
            denormalized_np = cv2.resize(denormalized_np, (resize_value, resize_value))
            
            cv2.imshow("Predicción - Original", face_display)
            cv2.imshow("Predicción - transformed", denormalized_np)
            cv2.waitKey(0)
                
        except Exception as e:
            print(f"Error procesando {path}: {str(e)}")
            continue
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Directorio de test images
    test_dir = file_path / "test_img_per"
    
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