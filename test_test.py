import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from my_loss.loss import FocalLoss, SSIM
import os
import numpy as np
import random

from torchvision import transforms
from dataset.data_loader_test import test_dataset
from tqdm import tqdm
from torchvision.utils import make_grid,save_image
import cv2
import gc
import albumentations as A

from PIL import Image
def trans(image):
    image = image.permute((1, 2, 0))
    image = image * 0.5 + 0.5
    image = np.array(image * 255, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)


def train_on_device(args):
    if not os.path.exists(args.data_save_path):
        os.makedirs(args.data_save_path)
    # 数据
    from dataset.data_txt_test import ReadConfigTxt
    images_list = ReadConfigTxt(args.data_path, update_txt=False, img_suffix=".jpg", split_str="||",
                                train_test_ratio=1.0)

    # transform = transforms.Compose([
    #     transforms.Resize(args.img_shape),
    #     transforms.ToTensor(),
    # ])
    transform = A.Compose([
        A.Resize(args.img_shape[0], args.img_shape[1]),
        A.RandomCrop(args.img_shape[0], args.img_shape[1]),
    ])
    dataset = test_dataset(data_root=args.data_path, image_list=images_list['all_patch'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.bs)

    # 网络定义
    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model_dict=torch.load('checkpoints/DRAEM_test_0.001_50_bs2_best.pckl')
    model.load_state_dict(model_dict)
    model.cuda()

    with torch.no_grad():
        for num,(image,name) in enumerate(tqdm(dataloader)):
            image=image.cuda()

            image_re = model(image)

            image_re = trans(image_re.cpu().squeeze())
            # path=args.data_save_path + name[0]
            cv2.imwrite(args.data_save_path+name[0],image_re)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--img_shape', type=int, default=[512, 512])

    # parser.add_argument('--data_path', type=str, default=r'E:\2021\dataset\earse\testA_resize_patch_512/')
    parser.add_argument('--data_path', type=str, default=r'E:\2021\dataset\earse2\testA_resize_patch_512/')
    # parser.add_argument('--data_save_path', type=str, default=r'E:\2021\dataset\earse\testA_resize_patch_512/earse_images/')
    parser.add_argument('--data_save_path', type=str,
                        default=r'E:\2021\dataset\earse2\testA_resize_patch_512/earse_images/')

    args = parser.parse_args()

    train_on_device(args)
