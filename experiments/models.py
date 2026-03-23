from torchvision import models
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained models - resnet
resnet18 = models.resnet18(pretrained=True).to(device)
resnet50 = models.resnet50(pretrained=True).to(device)
resnet101 = models.resnet101(pretrained=True).to(device)
resnet152 = models.resnet152(pretrained=True).to(device)

# Load pretrained models - vgg
vgg11 = models.vgg11(pretrained=True).to(device)
vgg13 = models.vgg13(pretrained=True).to(device)
vgg16 = models.vgg16(pretrained=True).to(device)
vgg19 = models.vgg19(pretrained=True).to(device)


# Load pretrained models - ConvNext
convnext_tiny = models.convnext_tiny(pretrained=True).to(device)
convnext_small = models.convnext_small(pretrained=True).to(device)
convnext_base = models.convnext_base(pretrained=True).to(device)
convnext_large = models.convnext_large(pretrained=True).to(device)


# etc.