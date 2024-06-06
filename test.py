import torch
from S4Fusion import *
from dataset import TestIVDataset
import logging
import tqdm
from torchvision import utils
import os
from torch import clamp
import modules.fusion

def YCrCb2RGB(Y, Cb, Cr):
    """
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out, min=0, max=1)
    return out

def test2(trained_model, ir_path, vi_path, save_path):
    dataset = TestIVDataset(ir_path=ir_path, vi_path=vi_path)
    
    model = MambaNet().to('cuda:1')
    file = torch.load(trained_model, map_location='cuda:1')
    model.load_state_dict(file['model'])
    model = model.eval()

    logging.info('all components are ready, start to test model...')
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset))):
            # input()
            vi, ir, left_1, right_1, left_2, right_2, name, Cb, Cr = dataset[i]
            # print(name)
            ir, vi = ir.unsqueeze(0).to('cuda:1'), vi.unsqueeze(0).to('cuda:1')
            fused_images = model(ir, vi)
            res = fused_images[0][:, left_1:right_1, left_2: right_2]
            utils.save_image(res.cpu(), \
                            os.path.join(save_path, name))
    print(modules.fusion.c_1 / len(dataset), modules.fusion.c_2 / len(dataset), \
          (modules.fusion.c_1 / len(dataset)) / (modules.fusion.c_2 / len(dataset)))
    print(modules.fusion.d_1 / len(dataset), modules.fusion.d_2 / len(dataset), \
          (modules.fusion.d_1 / len(dataset)) / (modules.fusion.d_2 / len(dataset)))

if __name__ == '__main__':
    test2(
        trained_model='./model/model.pkl',
        ir_path='',
        vi_path='',
        save_path='./results/'
    )