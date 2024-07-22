import torch
from UNet_resmine import Netv2,BottleNeck
def print_model_layers(model):
  for name, layer in model.named_modules():
    print(name)
net = Netv2(BottleNeck, [3, 4, 6, 3])
print_model_layers(net)