import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove last FC layer

    def forward(self, x):
        return self.feature_extractor(x)
