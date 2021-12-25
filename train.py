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
from dataset.data_loader2 import train_dataset
from tqdm import tqdm
from dataset.my_transform import *
import albumentations as A

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = 'DRAEM_test_' + str(args.lr) + '_' + str(args.epochs) + '_bs' + str(args.bs)
    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name + "/"))

    # 数据
    from dataset.data_txt import ReadConfigTxt2
    images_list = ReadConfigTxt2(args.data_path, update_txt=False, img_suffix=".jpg", split_str="||",
                                train_test_ratio=1.0)

    # transform = transforms.Compose([
    #     transforms.Resize(args.img_shape),
    #     # transforms.RandomAffine(degrees=(-5, 100), translate=(0, 0.05), scale=(0.8, 1.2), fillcolor=255),
    #     # transforms.CenterCrop(args.img_shape),
    #     transforms.ToTensor(),
    # ])
    # a=np.random.uniform(-5, 5, 1)
    # b=np.random.uniform(0.8,1.2,1)
    transform = A.Compose([
        A.Resize(args.img_shape[0],args.img_shape[1]),
        # A.ShiftScaleRotate(rotate_limit=np.random.uniform(-5,5,1),scale_limit=np.random.uniform(0.8,1.2,1),p=0.3),
        A.ShiftScaleRotate(rotate_limit=(-5,5),scale_limit=(-0.3,0.3), p=0.5),
        A.RandomCrop(args.img_shape[0],args.img_shape[1]),
    ])

    dataset = train_dataset(data_root=args.data_path, image_list=images_list['train_patch'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    # import cv2
    # from PIL import Image
    # def trans(image):
    #     image = image.permute((1, 2, 0))
    #     image = image * 0.5 + 0.5
    #     image = np.array(image * 255, dtype=np.uint8)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     return image
    # for image, gt in tqdm(dataset):
    #     image=trans(image)
    #     gt = trans(gt)
    #
    #     cv2.imshow('', image)
    #     cv2.imshow('0', gt)
    #     cv2.waitKey()

    # 网络定义
    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load('checkpoints/DRAEM_test_0.001_50_bs2_best.pckl'))
    model.cuda()
    model.apply(weights_init)


    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs * 0.8, args.epochs * 0.9], gamma=0.2,
                                               last_epoch=-1)
    from my_loss.my_loss import PSNR
    from my_loss.ssim2 import MS_SSIM
    # loss_MSE = torch.nn.modules.loss.MSELoss()
    # loss_ssim = SSIM()
    # loss_focal = FocalLoss().cuda()
    loss_psnr=PSNR().cuda()
    loss_ms_ssim=MS_SSIM(data_range=1.,channel=3,window_size=11).cuda()

    # torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl"))
    # torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name + "_seg.pckl"))

    n_iter = 0
    best_score=0
    best_score0 = 0
    for epoch in range(args.epochs):
        # print("Epoch: " + str(epoch))
        score_list=[]
        for image,gt in tqdm(dataloader,'train | epoch: %s | score: %s |'%(epoch,best_score)):
            image=image.cuda()
            gt = gt.cuda()

            image_rec = model(image)

            image_rec=image_rec*0.5+0.5
            gt=gt*0.5+0.5

            psnr_loss=loss_psnr(image_rec, gt)/100.0
            ms_ssim_loss = loss_ms_ssim(image_rec, gt).mean()

            loss = 1-psnr_loss + 1-ms_ssim_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            score = np.array(0.5 * psnr_loss.detach().cpu() + 0.5 * ms_ssim_loss.detach().cpu())
            score_list.append(score)

            if args.visualize and n_iter % 100 == 0:
                visualizer.plot_loss(psnr_loss, n_iter, loss_name='psnr')
                visualizer.plot_loss(ms_ssim_loss, n_iter, loss_name='ms_ssim')
                visualizer.plot_loss(score, n_iter, loss_name='score')

            n_iter += 1


            if best_score0 < score:
                best_score0 = score
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + "_best0.pckl"))
                # torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name + "_seg_best0.pckl"))

        if best_score<np.mean(np.array(score_list)):
            best_score=np.mean(np.array(score_list))
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + "_best.pckl"))
            # torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name + "_seg_best.pckl"))

        scheduler.step()

        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl"))
        # torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name + "_seg.pckl"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)  # 0.0001
    parser.add_argument('--epochs', type=int, default=50)  # 700
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--img_shape', type=int, default=[512, 512])

    parser.add_argument('--data_path', type=str, default=r'E:\2021\dataset\earse\train_resize_patch_512/')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--visualize', type=bool, default=True)

    args = parser.parse_args()

    # text
    # if not os.path.exists(args.data_path+'train.txt'):
    # # if True:
    #     from dataset.data_txt import GenConfigTxt
    #     GenConfigTxt(args.data_path, img_suffix=".jpg", split_str="||", train_test_ratio=1.0)
    #

    train_on_device(args)
