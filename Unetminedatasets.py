import glob
import logging
from PIL import Image
from torch.utils.data import DataLoader as Dataloader
from torch.utils.data import Dataset as Dataset
from torchvision import transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import splitext, isfile, join
from tqdm import tqdm
from pathlib import Path
# imgdir = 'D:\chenzhehao\work\Pytorch-UNet-master\Pytorch-UNet-master\data\imgs'
# maskdir = 'D:\chenzhehao\work\Pytorch-UNet-master\Pytorch-UNet-master\data\masks'
# img1 = 'D:\chenzhehao\work\Pytorch-UNet-master\Pytorch-UNet-master\data\masks/8.png'

def load_image(filename):   #直接打开图片文件，没有问题
     return Image.open(filename)#.convert('L')      #不适用L灰度模式，mask——values为【0，1，2】，使用L后变为【0，38，72】

# img = load_image(img1)
# plt.figure("Image") # 图像窗口名称
# plt.imshow(img)
# plt.show()
def unique_mask_values(idx, mask_dir, mask_suffix):    #对mask图片进行处理
    mask_file = glob.glob(str(mask_dir)+'/'+idx + mask_suffix + '.*')
    mask_file = ''.join(mask_file)
    #print('maskfile is',mask_file)
    mask = np.asarray(load_image(mask_file))
    #print('mask is',mask.shape )
    #print('ndim is',mask.ndim)
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        #print('shape is',mask.shape)
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

# unique = []
# ids = [splitext(file)[0] for file in listdir(imgdir) if        #没问题，ids为['1','2',.....]
#                     isfile(join(imgdir, file)) and not file.startswith('.')]
# print('ids is', ids)
# for id in tqdm(ids):
#      #result = unique_mask_values(id, mask_dir=maskdir, mask_suffix='')
#      result = unique_mask_values(id, maskdir, mask_suffix='')
#      #result = ''.join(result)
#      print('result is',result)
#      unique.append(result)
#
# print('unique is', unique)
# mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
# print('maskvalue is',mask_values)
# 创建数据集
class UnetmineDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, images_dir: str, mask_dir: str, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix
        self.new_w = 512
        self.new_h = 512

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        unique = []

        for id in tqdm(self.ids):          #  unique列表存的是各个mask图片名，组成一个列表
            result = unique_mask_values(id, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix)
            #result = glob.glob(maskdir + '/' + id + mask_suffix + '.*')
            #result = ''.join(result)
            unique.append(result)


        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))  #感觉这句没啥用，maskvalues返回的也是列表
        print('self.mask_values',self.mask_values)
        logging.info(f'Unique mask values: {self.mask_values}')


    def __len__(self):
        return len(self.ids)


    @staticmethod
    def preprocess(mask_values, pil_img, new_w, new_h, is_mask):
        w, h = pil_img.size
        newW, newH = int(new_w), int(new_h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)

        if is_mask:  #如果是mask图片，根据 mask_values 列表中的像素值，将掩模数组中对应像素位置的值设置为 i。返回掩模数组 mask。
            img = np.array(pil_img)
            mask = np.zeros((newH, newW), dtype=np.int64)
            #print('maskshape_is',mask.shape)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            img = np.array(pil_img)
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])


        # assert img.size == mask.size, \
        #     f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.new_w, self.new_h, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.new_w, self.new_h, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


# imgdir = 'F:/picture/testdataset/images'
# maskdir = 'F:/picture/testdataset/labels'
# # img = Image.open(img1).convert('L')
# a = UnetmineDataset(imgdir,maskdir,'')
# b = a[0]['image']
# c = a[0]['mask']
# print(b.shape)
# print(c.shape)
#np.savetxt('array.txt', b, delimiter=',')
# with open('data.txt', 'w') as file:
#     # 遍历字典并将键和值写入文本文件
#     for key, value in b.items():
#         file.write(f'{key}: {value}\n')