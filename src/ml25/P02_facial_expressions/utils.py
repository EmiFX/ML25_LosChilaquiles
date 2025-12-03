import numpy as np
import os
import json
import pathlib
import cv2
import torch
import torch.nn as nn
import torchvision

file_path = pathlib.Path(__file__).parent.absolute()

def enhance_image(img: np.ndarray, method="clahe"):
    """
    Mejora la calidad de imágenes en escala de grises
    
    Args:
        img (np.ndarray): Imagen uint8 (0-255)
        method (str): 'clahe', 'full', o 'none'
    
    Returns:
        np.ndarray: Imagen mejorada
    """
    if method == "none":
        return img
    
    # Asegurar uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    if method == "clahe":
        # Solo CLAHE (rápido y efectivo)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    
    elif method == "full":
        # Pipeline completo (más lento pero mejor calidad)
        # 1. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        
        # 2. Denoising
        enhanced = cv2.fastNlMeansDenoising(enhanced, None, h=10)
        
        # 3. Sharpening suave
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]]) / 9
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Combinar
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        return result
    
    return img

class EnhanceTransform:
    """Transformación personalizada para mejorar imágenes"""
    def __init__(self, method="clahe"):
        self.method = method
    
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return enhance_image(img, method=self.method)
        return img


def get_transforms(split, img_size, enhance=True):
    # El dataset consiste en imagenes en escala de grises
    # con valores entre 0 y 255
    # de dimension 1 x 48 x 48
    # TODO: Define las trasnformaciones para el conjunto de entrenamiento y validacion
    # Agrega algún tipo de data agumentation para el conjunto de entrenamiento
    # https://pytorch.org/vision/stable/transforms.html
    common = [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((img_size, img_size)),
    ]

    mean, std = 0.5, 0.5

    if enhance:
        base = [EnhanceTransform(method="clahe")] + common

    if split == "train":
        transforms = torchvision.transforms.Compose(
            [
                *common,
                torchvision.transforms.ColorJitter(
                    brightness=0.5, contrast=0.4, saturation=0, hue=0
                ),
                torchvision.transforms.Normalize((mean,), (std,)),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [*common, torchvision.transforms.Normalize((mean,), (std,))]
        )

    # For visualization
    deNormalize = UnNormalize(mean=[mean], std=[std])
    return transforms, deNormalize


# Definir las transformaciones segun el conjunto
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def to_torch(array: np.ndarray, roll_dims=True):
    """
    Convert tensor to numpy array
    args:
        - array (np.ndarray): array to convert
            size: (H, W, C)
    returns:
        - array (np.ndarray): converted tensor
            size: (C, H, W)
    """
    if roll_dims:
        if len(array.shape) <= 2:
            array = np.expand_dims(array, axis=2)  # (H, W) -> (H, W, 1)
        array = array.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
    tensor = torch.tensor(array)
    return tensor


def to_numpy(tensor: torch.Tensor, roll_dims=True):
    """
    Convert tensor to numpy array
    args:
        - tensor (torch.tensor): tensor to convert
            size: (C, H, W)
    returns:
        - array (np.ndarray): converted array
            size: (H, W, C)
    """
    if roll_dims:
        if len(tensor.shape) > 3:
            tensor = tensor.squeeze(0)  # (1, C, H, W) -> (C, H, W)
        tensor = tensor.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    array = tensor.detach().cpu().numpy()
    return array


def add_img_text(img: np.ndarray, text_label: str):
    """
    Add text to image
    args:
        - img (np.ndarray): image to add text to
            - size: (C, H, W)
        - text (str): text to add to image
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 0, 0)
    thickness = 2

    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    (text_w, text_h), _ = cv2.getTextSize(text_label, font, fontScale, thickness)

    # Center text
    x1, y1 = 0, text_h  # Top left corner
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), (255, 255, 255), -1)
    if img.shape[-1] == 1 or len(img.shape) == 2:  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.putText(img, text_label, (x1, y1), font, fontScale, fontColor, thickness)
    return img


def create_train_val_split():
    # Usado para crear split.json
    import pandas as pd

    train_csv = file_path / "data/train.csv"
    df = pd.read_csv(train_csv)
    n_samples = len(df)
    val_samples = np.random.choice(n_samples, size=int(n_samples // 5), replace=False)
    train_samples = np.setdiff1d(np.arange(n_samples), val_samples)
    sample_dct = {"train": train_samples.tolist(), "val": val_samples.tolist()}
    outfile = file_path / "data/split.json"
    with open(outfile, "w") as f:
        json.dump(sample_dct, f, indent=2)
