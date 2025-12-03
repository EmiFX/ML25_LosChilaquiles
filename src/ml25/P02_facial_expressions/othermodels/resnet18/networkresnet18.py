#network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import resnet18, ResNet18_Weights

file_path = pathlib.Path(__file__).parent.absolute()

#modelo preentrenado, resnet18
def build_backbone(model="resnet18", weights="imagenet", freeze=True, last_n_layers=2):
    if model == "resnet18":
        if weights == "imagenet":
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)

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
        self.backbone = build_backbone(model="resnet18", weights="imagenet", freeze=True, last_n_layers=2)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # TODO: Calcular dimension de salida
        #out_dim = self.calc_out_dim(input_dim, kernel_size=3, stride=1, padding=1)
        #out_dim = self.calc_out_dim(out_dim, kernel_size=2, stride=2, padding=0)  # maxpool
        #out_dim = self.calc_out_dim(out_dim, kernel_size=3, stride=1, padding=1)
        #out_dim = self.calc_out_dim(out_dim, kernel_size=2, stride=2, padding=0)  # maxpool
        
        # TODO: Define las capas de tu red
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Global Average Pooling para convertir features espaciales a vector
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Capas fully connected finales
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)
        
        # Adaptar la primera capa del backbone para recibir 1 canal (grayscale)
        # en vez de 3 canales (RGB)
        original_conv1 = self.backbone[0]
        self.backbone[0] = nn.Conv2d(
            1,  # 1 canal de entrada (grayscale)
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        # Promediar los pesos de los 3 canales RGB al 1 canal grayscale
        with torch.no_grad():
            self.backbone[0].weight = nn.Parameter(
                original_conv1.weight.mean(dim=1, keepdim=True)
           )
        
        self.to(self.device)
    
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2 * padding) / stride) + 1
        return out_dim
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Define la propagacion hacia adelante de tu red
        x = self.backbone(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu((self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x)
        x=x.view(x.size(0), -1)
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
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