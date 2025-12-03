from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from ml25.P02_facial_expressions.dataset import get_loader, EMOTIONS_MAP
from ml25.P02_facial_expressions.othermodels.resnet18.networkresnet18 import Network
from collections import Counter

# Logging
import wandb
from datetime import datetime, timezone

def calculate_class_weights(dataset):
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase
    """
    # Contar cuántas muestras hay de cada emoción
    labels = [sample['label'] for sample in dataset]
    class_counts = Counter(labels)
    
    # Total de muestras
    total = len(labels)
    
    # Calcular peso: total / (n_classes * count_per_class)
    n_classes = len(class_counts)
    class_weights = {}
    
    for emotion_id, count in class_counts.items():
        weight = total / (n_classes * count)
        class_weights[emotion_id] = weight
    
    # Convertir a tensor ordenado [0, 1, 2, 3, 4, 5, 6]
    weights_list = [class_weights[i] for i in range(n_classes)]
    weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
    
    print("\n📊 Distribución de clases:")
    for emotion_id, count in sorted(class_counts.items()):
        emotion_name = EMOTIONS_MAP[emotion_id]
        weight = class_weights[emotion_id]
        print(f"  {emotion_name}: {count} samples (weight: {weight:.3f})")
    
    return weights_tensor

def init_wandb(cfg):
    # Initialize wandb
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H-%M-%S-%f")

    run = wandb.init(
        project="facial_expressions_cnn",
        config=cfg,
        name=f"facial_expressions_cnn_{timestamp}_utc",
    )
    return run


def validation_step(val_loader, net, cost_function):
    """
    Realiza un epoch completo en el conjunto de validación
    args:
    - val_loader (torch.DataLoader): dataloader para los datos de validación
    - net: instancia de red neuronal de clase Network
    - cost_function (torch.nn): Función de costo a utilizar

    returns:
    - val_loss (float): el costo total (promedio por minibatch) de todos los datos de validación
    """
    val_loss = 0.0
    device = net.device
    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch["transformed"].to(device)
        batch_labels = batch["label"].to(device)
        device = net.device
        batch_labels = batch_labels.to(device)
        with torch.inference_mode():
            # TODO: realiza un forward pass, calcula el loss y acumula el costo
            logits, proba = net(batch_imgs)
            loss = cost_function(logits, batch_labels)
            val_loss += loss.item()
            
    # TODO: Regresa el costo promedio por minibatch
    return val_loss/len(val_loader)


def train():
    # Hyperparametros
    cfg = {
        "training": {
            "learning_rate": 5e-3,
            "n_epochs": 50,
            "batch_size": 1024,
        },
    }
    run = init_wandb(cfg)

    train_cfg = cfg.get("training", {})
    learning_rate = train_cfg.get("learning_rate", 1e-4)
    n_epochs = train_cfg.get("n_epochs", 100)
    batch_size = train_cfg.get("batch_size", 256)

    # Train, validation, test loaders
    train_dataset, train_loader = get_loader(
        "train", batch_size=batch_size, shuffle=True
    )
    val_dataset, val_loader = get_loader("val", batch_size=batch_size, shuffle=False)

    print(
        f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}"
    )

    # Instanciamos tu red
    modelo = Network(input_dim=48, n_classes=7)

    class_weights = calculate_class_weights(train_dataset)
    device = modelo.device
    class_weights = class_weights.to(device)

    # TODO: Define la funcion de costo
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Define el optimizador
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate)

    best_epoch_loss = np.inf
    #device = modelo.device
    for epoch in range(n_epochs):
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch["transformed"].to(device)
            batch_labels = batch["label"].to(device)
            # TODO Zero grad, forward pass, backward pass, optimizer step
            optimizer.zero_grad()
            logits, proba = modelo(batch_imgs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            # TODO acumula el costo
            train_loss += loss.item()

        # TODO Calcula el costo promedio
        train_loss = train_loss / len(train_loader)
        val_loss = validation_step(val_loader, modelo, criterion)
        tqdm.write(
            f"Epoch: {epoch}, train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}"
        )

        # TODO guarda el modelo si el costo de validación es menor al mejor costo de validación
        if val_loss < best_epoch_loss:
            best_epoch_loss = val_loss
            modelo.save_model("best_model.pth")
            tqdm.write(f"Modelo guardado en epoch {epoch} con val_loss: {val_loss:.2f}")


        run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
            }
        )


if __name__ == "__main__":
    train()