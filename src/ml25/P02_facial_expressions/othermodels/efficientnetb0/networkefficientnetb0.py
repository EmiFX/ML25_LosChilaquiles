#network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

file_path = pathlib.Path(__file__).parent.absolute()

#modelo preentrenado, efficientnet_b0
def build_backbone(model="efficientnet_b0", weights="imagenet", freeze=True, last_n_layers=2):
    if model == "efficientnet_b0":
        if weights == "imagenet":
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            backbone = efficientnet_b0(weights=None)

        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        
            # Descongelar últimos bloques de EfficientNet
            # EfficientNet tiene 8 bloques (features[0] a features[8])
            if last_n_layers >= 1:
                for param in backbone.features[-1].parameters():
                    param.requires_grad = True
            if last_n_layers >= 2:
                for param in backbone.features[-2].parameters():
                    param.requires_grad = True
            if last_n_layers >= 3:
                for param in backbone.features[-3].parameters():
                    param.requires_grad = True               
        return backbone
    else:
        raise Exception(f"Model {model} not supported")

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.backbone = build_backbone(model="efficientnet_b0", weights="imagenet", freeze=True, last_n_layers=2)
        
        # TODO: Calcular dimension de salida
        backbone_out_features = 1280  # EfficientNet-B0 tiene 1280 features de salida
        self.backbone.classifier = nn.Identity()  # para borrar la ultima capa de efficientnet

        # TODO: Define las capas de tu red
        self.fc1 = nn.Linear(backbone_out_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, n_classes)
        
        self.to(self.device)
    
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2 * padding) / stride) + 1
        return out_dim
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Define la propagacion hacia adelante de tu red
        # escala a rgb x si es necesario
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Extraer features con EfficientNet-B0
        x = self.backbone(x)  # (batch_size, 1280)
        
        x = self.fc1(x)  # 1280 -> 512
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        logits = self.fc2(x)
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