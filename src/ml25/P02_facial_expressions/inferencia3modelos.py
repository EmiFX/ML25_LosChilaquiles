import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import cv2
import torch
from ml25.P02_facial_expressions.othermodels.resnet18.networkresnet18 import Network as ResNet18Network
from ml25.P02_facial_expressions.othermodels.resnet50.networkresnet50 import Network as ResNet50Network
from ml25.P02_facial_expressions.othermodels.efficientnetb0.networkefficientnetb0 import Network as EfficientNetB0Network
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


def load_models():
    """
    Carga los 3 modelos entrenados
    returns:
    - dict con los modelos cargados
    """
    models = {}
    
    # Modelo ResNet18
    try:
        modelo_resnet18 = ResNet18Network(48, 7)
        modelo_resnet18.load_model("best_model_resnet18.pth")
        modelo_resnet18.eval()
        models['ResNet18'] = modelo_resnet18
        print("✓ ResNet18 cargado")
    except Exception as e:
        print(f"✗ Error cargando ResNet18: {e}")
    
    # Modelo ResNet50
    try:
        modelo_resnet = ResNet50Network(48, 7)
        modelo_resnet.load_model("best_model_resnet50.pth")
        modelo_resnet.eval()
        models['ResNet50'] = modelo_resnet
        print("✓ ResNet50 cargado")
    except Exception as e:
        print(f"✗ Error cargando ResNet50: {e}")
    
    # Modelo EfficientNet-B0
    try:
        modelo_efficient = EfficientNetB0Network(48, 7)
        modelo_efficient.load_model("best_model_efficientnetb0.pth")
        modelo_efficient.eval()
        models['EfficientNet-B0'] = modelo_efficient
        print("✓ EfficientNet-B0 cargado")
    except Exception as e:
        print(f"✗ Error cargando EfficientNet-B0: {e}")
    
    return models


def predict_with_all_models(img_paths):
    """
    Hace inferencia con todos los modelos disponibles
    args:
    - img_paths (list): lista de paths a las imagenes
    """
    # Cargar todos los modelos
    models = load_models()
    
    if not models:
        print("No se pudo cargar ningún modelo")
        return
    
    print(f"\n{'='*60}")
    print(f"Modelos cargados: {', '.join(models.keys())}")
    print(f"{'='*60}\n")
    
    for path in img_paths:
        print(f"\nProcesando: {Path(path).name}")
        print("-" * 40)
        
        # Cargar la imagen
        im_file = (file_path / path).as_posix() if not os.path.isabs(path) else path
        original, transformed, denormalized = load_img(im_file)
        
        # Preparar imagen para visualización
        h, w = original.shape[:2]
        resize_value = 300
        img_display = cv2.resize(original, (w * resize_value // h, resize_value))
        
        # Inferencia con cada modelo
        predictions = {}
        for model_name, modelo in models.items():
            device = modelo.device
            transformed_batch = transformed.unsqueeze(0).to(device)
            
            with torch.inference_mode():
                logits, proba = modelo.predict(transformed_batch)
                pred = torch.argmax(proba, -1).item()
                pred_label = EMOTIONS_MAP[pred]
                confidence = proba[0, pred].item() * 100
                
                predictions[model_name] = {
                    'label': pred_label,
                    'confidence': confidence,
                    'proba': proba[0].cpu().numpy()
                }
                
                print(f"{model_name:20s}: {pred_label:10s} ({confidence:5.2f}%)")
        
        # Crear visualización comparativa
        img_with_preds = img_display.copy()
        y_offset = 30
        
        for i, (model_name, pred_info) in enumerate(predictions.items()):
            text = f"{model_name}: {pred_info['label']} ({pred_info['confidence']:.1f}%)"
            color = (0, 255, 0) if i == 0 else (255, 255, 0) if i == 1 else (255, 0, 255)
            cv2.putText(img_with_preds, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        # Mostrar imágenes
        denormalized_np = to_numpy(denormalized)
        denormalized_np = cv2.resize(denormalized_np, (resize_value, resize_value))
        
        cv2.imshow("Predicciones - Original", img_with_preds)
        cv2.imshow("Predicciones - Transformada", denormalized_np)
        
        key = cv2.waitKey(0)
        if key == 27:  # ESC para salir
            break
    
    cv2.destroyAllWindows()


def predict_single_model(img_paths, model_type='resnet50'):
    """
    Hace inferencia con un solo modelo específico
    args:
    - img_paths (list): lista de paths a las imagenes
    - model_type (str): 'resnet18', 'resnet50', o 'efficientnetb0'
    """
    # Cargar el modelo especificado
    if model_type == 'resnet18':
        modelo = ResNet18Network(48, 7)
        modelo.load_model("best_model_resnet18.pth")
        model_name = "ResNet18"
    elif model_type == 'resnet50':
        modelo = ResNet50Network(48, 7)
        modelo.load_model("best_model_resnet50.pth")
        model_name = "ResNet50"
    elif model_type == 'efficientnetb0':
        modelo = EfficientNetB0Network(48, 7)
        modelo.load_model("best_model_efficientnetb0.pth")
        model_name = "EfficientNet-B0"
    else:
        raise ValueError(f"model_type debe ser 'resnet18', 'resnet50' o 'efficientnetb0', no '{model_type}'")
    
    modelo.eval()
    device = modelo.device
    
    print(f"\nUsando modelo: {model_name}")
    print(f"{'='*60}\n")
    
    for path in img_paths:
        # Cargar la imagen
        im_file = (file_path / path).as_posix() if not os.path.isabs(path) else path
        original, transformed, denormalized = load_img(im_file)
        
        transformed_batch = transformed.unsqueeze(0).to(device)
        
        # Inferencia
        with torch.inference_mode():
            logits, proba = modelo.predict(transformed_batch)
            pred = torch.argmax(proba, -1).item()
            pred_label = EMOTIONS_MAP[pred]
            confidence = proba[0, pred].item() * 100
        
        # Visualización
        h, w = original.shape[:2]
        resize_value = 300
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        img = add_img_text(img, f"{model_name} - {pred_label} ({confidence:.1f}%)")
        
        denormalized_np = to_numpy(denormalized)
        denormalized_np = cv2.resize(denormalized_np, (resize_value, resize_value))
        
        cv2.imshow("Predicción - Original", img)
        cv2.imshow("Predicción - Transformada", denormalized_np)
        
        key = cv2.waitKey(0)
        if key == 27:  # ESC para salir
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Directorio de test images
    test_dir = file_path / "test_imgs"
    
    # Get all image files
    img_paths = [str(p) for p in test_dir.iterdir() 
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not img_paths:
        print("No se encontraron imágenes en test_imgs/")
        exit(1)
    
    print("\n" + "="*60)
    print("INFERENCIA - RECONOCIMIENTO DE EXPRESIONES FACIALES")
    print("="*60)
    print("\nSelecciona el modelo a usar:")
    print("  1. ResNet18")
    print("  2. ResNet50")
    print("  3. EfficientNet-B0")
    print("  4. Comparar todos los modelos")
    print("="*60)
    
    try:
        opcion = input("\nIngresa el número (1-4): ").strip()
        
        if opcion == '1':
            predict_single_model(img_paths, model_type='resnet18')
        elif opcion == '2':
            predict_single_model(img_paths, model_type='resnet50')
        elif opcion == '3':
            predict_single_model(img_paths, model_type='efficientnetb0')
        elif opcion == '4':
            predict_with_all_models(img_paths)
        else:
            print("\n❌ Opción inválida. Usa 1, 2, 3 o 4")
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")