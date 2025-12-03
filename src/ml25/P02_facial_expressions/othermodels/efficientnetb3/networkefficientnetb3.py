#network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import resnet50, ResNet50_Weights

file_path = pathlib.Path(__file__).parent.absolute()

#modelo preentrenado, resnet50
def build_backbone(model="resnet50", weights="imagenet", freeze=True, last_n_layers=2):
    if model == "resnet50":
        if weights == "imagenet":
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet50(weights=None)

        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone
    else:
        raise Exception(f"Model {model} not supported")


class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.backbone = build_backbone(model="resnet50", weights="imagenet", freeze=True, last_n_layers=2)

        backbone_out_features = 2048 #resnet50 tiene 2048 features de salida
        self.backbone.fc = nn.Identity() #para borrar la ultima capa de resnet

        # TODO: Calcular dimension de salida
        
        # TODO: Define las capas de tu red
        self.fc1 = nn.Linear(backbone_out_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, n_classes)
        
        self.to(self.device)
    
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2 * padding) / stride) + 1
        return out_dim
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # escala a rgb x si es necesario
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # TODO: Define la propagacion hacia adelante de tu red
        x = self.backbone(x) #batch, 2048
        
        x = self.fc1(x) #2048, 512
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        logits = self.fc3(x)
        proba = F.softmax(logits, dim=1)
        return logits, proba
    
    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)
    
    def save_model(self, model_name: str):
        """
        Guarda el modelo en el path especificado
        args:
        - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
        - path (str): path relativo donde se guardará el modelo
        """
        models_path = file_path / "models" / model_name
        if not models_path.parent.exists():
            models_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(self.state_dict(), models_path)
    
    def load_model(self, model_name: str):
        """
        Carga el modelo en el path especificado
        args:
        - path (str): path relativo donde se guardó el modelo
        """
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / "models" / model_name
        self.load_state_dict(torch.load(models_path, map_location=self.device))
        self.to(self.device)