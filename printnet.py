import torch
from U_net_resbasic_slimconv_AAM_CA import Netv3,BasicBlock
def print_model_layers(model):
  for name, layer in model.named_modules():
    print(name)
net = Netv3(BasicBlock, [1,1,1,1])
print_model_layers(net)