import numpy as np
import math
import torch
import torch.nn as nn

def psnr1(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2)
   # if mse < 1.0e-10:
   #    return 100
   return 10 * math.log10(255.0**2/mse)

def psnr(image, gt):
   loss_MSE=torch.nn.modules.loss.MSELoss()
   MSE_loss=loss_MSE(image, gt)
   return 10 * math.log10(255.0**2/MSE_loss)

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        self.loss = torch.nn.modules.loss.MSELoss()

    def forward(self,image, gt):
        loss = self.loss(image, gt)
        return 10.0 * torch.log10(1.0 ** 2.0 / loss)

