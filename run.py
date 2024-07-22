import os
import PIL
import numpy
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image

from utils import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp

from Unetminedatasets import UnetmineDataset
from U_net_resbasic_slimconv_AAM_CA import Netv3,BasicBlock
from PIL import Image
img_dir = 'D:/chenzhehao/git-repos/gradcam'

img_name1 = 'Image__2024-05-21__21-06-21.bmp'
img_name2 = 'Image__2024-05-21__21-08-10.bmp'
img_path = [os.path.join(img_dir, img_name1),os.path.join(img_dir, img_name1)]
full_img1= Image.open(img_path[0])
full_img2= Image.open(img_path[1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#process

img1 = torch.from_numpy(UnetmineDataset.preprocess([0,1,2], full_img1, 512, 512, False))
img2 = torch.from_numpy(UnetmineDataset.preprocess([0,1,2], full_img2, 512, 512, False))
img =torch.cat((img1,img2),0)
img = img.unsqueeze(1)
img = img.to(device=device, dtype=torch.float32)
#部署模型

net = Netv3(BasicBlock, [1,1,1,1])

net.to(device=device)
state_dict =torch.load('D:/chenzhehao/git-repos/gradcam/UNet+res +AAM+slimconv+CA 1111   basicblock.pth', map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1, 2])
net.load_state_dict(state_dict)
#一系列模型的cam和cam++
cam_dict = dict()
model_dict = dict(type='RAUnet', arch=net, layer_name='decoder4.deconv2',input_size=(512, 512))
gradcam = GradCAM(model_dict,class_idx = 1, retain_graph = True)
gradcampp = GradCAMpp(model_dict,class_idx = 1, retain_graph = True)
cam_dict['net'] = [gradcam, gradcampp]
# cam_dict['net'] = [gradcam]


#计算
images = []
for gradcam, gradcam_pp in cam_dict.values():
# for gradcam in cam_dict.values():
    mask, _ = gradcam(img)
    heatmap, result = visualize_cam(mask, img)

    mask_pp, _ = gradcam_pp(img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, img)

    images.append(torch.stack([img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
    # images.append(torch.stack([img.squeeze().cpu(), heatmap, result,], 0))

images = make_grid(torch.cat(images, 0), nrow=5)


#显示结果
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
output_name = img_name1+img_name2
output_path = os.path.join(output_dir, output_name)

save_image(images, output_path)
PIL.Image.open(output_path)