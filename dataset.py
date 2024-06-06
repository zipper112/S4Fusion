import torch
from PIL import Image
from torchvision import transforms
import os
from torch.nn import functional as F
from torch import clamp
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def RGB2YCrCb(rgb_image):

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y, min=0, max=1)
    Cr = clamp(Cr, min=0, max=1)
    Cb = clamp(Cb, min=0, max=1)
    return Y, Cb, Cr

class FuseDataset:
    def __init__(self, vi_path: str, ir_path: str, crop_size: int) -> None:
        self.image_path = vi_path
        self.label_path = ir_path
        self.crop_size = crop_size

        self.vi = [os.path.join(vi_path, imgname) for imgname in sorted(os.listdir(vi_path))]
        self.ir = [os.path.join(ir_path, imgname) for imgname in sorted(os.listdir(ir_path))]

        self.totensor = transforms.ToTensor()
        self.flip1 = transforms.RandomHorizontalFlip()
        self.flip2 = transforms.RandomVerticalFlip()
        self.croper = transforms.RandomCrop(crop_size)

    def __len__(self):
        return len(self.vi)

    def __getitem__(self, index: int):
        vi = Image.open(self.vi[index]).convert('RGB')
        ir = Image.open(self.ir[index]).convert('L')
        factor = 1
        while vi.size[0] < self.crop_size[0] or vi.size[1] < self.crop_size[1]:
            vi = Image.open(self.vi[(index + factor) % len(self)]).convert('RGB')
            ir = Image.open(self.ir[(index + factor) % len(self)]).convert('L')
            factor += 1

        vi = self.totensor(vi)
        ir = self.totensor(ir)

        img = torch.cat([vi, ir], dim=0)
        img = self.croper(img)

        img = self.flip1(self.flip2(img))

        vi, ir = img[:3, :, :], img[3:, :, :]

        return vi, ir


class TestIVDataset:
    def __init__(self, vi_path, ir_path) -> None:
        self.names = sorted(os.listdir(vi_path))
        self.vi = [os.path.join(vi_path, imgname) for imgname in sorted(os.listdir(vi_path))]
        self.ir = [os.path.join(ir_path, imgname) for imgname in sorted(os.listdir(ir_path))]

        self.totensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.vi)
    
    def get_rest(self, x):
        add = 0
        while (x - 1) % 3 != 0 or ((x - 1) // 3) % 4 != 0:
            add += 1
            x += 1
        # add = 16 - x % 16
        return add

    def compute_pad(self, x):
        rest = self.get_rest(x)
        left = rest // 2
        return left, rest - left, left, left + x

    def __getitem__(self, index):
        Y, Cb, Cr = RGB2YCrCb(self.totensor(Image.open(self.vi[index]).convert('RGB')))
        vi = Y
        ir = self.totensor(Image.open(self.ir[index]).convert('L'))

        left_1_pad, right_1_pad, left_1, right_1 = self.compute_pad(vi.shape[1])
        left_2_pad, right_2_pad, left_2, right_2 = self.compute_pad(vi.shape[2])

        vi, ir = F.pad(vi, (left_2_pad, right_2_pad, left_1_pad, right_1_pad), "constant", 0.) ,\
            F.pad(ir, (left_2_pad, right_2_pad, left_1_pad, right_1_pad), "constant", 0.)

        return vi, ir, left_1, right_1, left_2, right_2, self.names[index], Cb, Cr