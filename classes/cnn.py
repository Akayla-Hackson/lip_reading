import torchvision.models as models
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove last FC layer

    def forward(self, x):
        return self.feature_extractor(x)
