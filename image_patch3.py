import cv2
import numpy as np
from glob import glob
import torch
from einops import repeat
from tqdm import tqdm
import os


# 自动判断剪贴比例
def cut_mode(h, w):
    h_list = [1, 2, 3, 4]
    w_list = [1, 2, 3, 4]
    h_list = torch.tensor(h_list, dtype=torch.float32)
    w_list = torch.tensor(w_list, dtype=torch.float32)

    # 各种假设比例
    con_val = []
    for i in h_list:
        for j in w_list:
            if i == j:
                i, j = 1, 1
            if i == 2 * j:
                i, j = 1, 2
            if 2 * i == j:
                i, j = 2, 1
            con_val.append([i, j, i * 1.0 / j])
    con_val = torch.tensor(con_val, dtype=torch.float32)

    # 原图比例
    ratio = torch.tensor(h, dtype=torch.float32) / torch.tensor(w, dtype=torch.float32)
    ratio = [ratio for i in range(len(con_val))]
    ratio = torch.tensor(ratio)

    # 计算最小距离
    dist = torch.cdist(ratio.unsqueeze(1), con_val[:, 2].unsqueeze(1))
    min_idx = torch.argmin(dist)

    # 分割方案
    mode = np.array([con_val[min_idx][0], con_val[min_idx][1]], dtype=np.int16)

    # 按照原图分辨率扩大比例
    if mode[0] == mode[1] or mode[0] * 2 == mode[1] or mode[0] == mode[1] * 2:
        if h > 3400 and h < 9999 and w > 3400 and w < 9999:
            # mode=[con_val[min_idx][0]*4,con_val[min_idx][1]*4]
            mode *= 4
        if h > 1500 and h < 3400 and w > 1500 and w < 3400:
            # mode=[con_val[min_idx][0]*3,con_val[min_idx][1]*3]
            mode *= 3
        if h > 1000 and h < 1500 and w > 1000 and w < 1500:
            # mode=[con_val[min_idx][0]*2,con_val[min_idx][1]*2]
            mode *= 2
    else:
        pass

    return mode


# 剪切图片
def crop(dataroot=None, stride=256,save_txt_path=None):
    # save_txt_path = r"E:\2021\dataset\earse\train_resize_patch_%s" % str(stride)
    image_list = ReadConfigTxt(dataroot, img_suffix=".jpg", save_path=save_txt_path)

    # generate path
    image_save_path = r"E:\2021\dataset\earse\train_resize_%s\images" % str(stride)
    image_gt_save_path = r"E:\2021\dataset\earse\train_resize_%s\gts" % str(stride)
    mask_save_path = r"E:\2021\dataset\earse\train_resize_%s\mask" % str(stride)

    image_save_path2 = r"E:\2021\dataset\earse\train_resize_patch_%s\images" % str(stride)
    image_gt_save_path2 = r"E:\2021\dataset\earse\train_resize_patch_%s\gts" % str(stride)
    mask_save_path2 = r"E:\2021\dataset\earse\train_resize_patch_%s\mask" % str(stride)
    path_list = [image_save_path, image_gt_save_path, mask_save_path, image_save_path2, image_gt_save_path2,
                 mask_save_path2]
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)

    phases = ["train", "val"]

    for phases0 in phases:
        with open(os.path.join(save_txt_path, "{}_patch.txt".format(phases0)), mode='w') as f:
            # for i, img_path in enumerate(tqdm(image_paths)):
            for i, img_path in enumerate(tqdm(image_list[phases0])):
                # image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                image = cv2.imread(img_path)

                # 判断分割策略；比例
                H, W = image.shape[:2]
                mode = cut_mode(H, W)
                new_W = mode[1] * stride
                new_H = mode[0] * stride

                # image
                image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_AREA)

                # gt
                image_gt_paths = img_path.replace("image", "gt").replace("jpg", "png")
                # image_gt = cv2.imdecode(np.fromfile(image_gt_paths, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                image_gt = cv2.imread(image_gt_paths)
                image_gt = cv2.resize(image_gt, (new_W, new_H), interpolation=cv2.INTER_AREA)

                # mask
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_gt_gray = cv2.cvtColor(image_gt, cv2.COLOR_BGR2GRAY)
                image_sub = cv2.subtract(image_gt_gray, image_gray)
                retval, mask = cv2.threshold(image_sub, 35, 255, cv2.THRESH_BINARY)

                # save reshape image
                file_name = img_path.split("images")[-1]  # '\\dehw_train_00000.jpg'
                cv2.imwrite(image_save_path + file_name, image)
                cv2.imwrite(image_gt_save_path + file_name, image_gt)
                cv2.imwrite(mask_save_path + file_name, mask)

                index = 0
                for col in range(mode[1]):
                    for row in range(mode[0]):
                        name = img_path.split("images")[-1].replace(".", "_" + str(index) + ".")

                        image_Patch = image[stride * row:stride * (row + 1), stride * col:stride * (col + 1), :]
                        image_gt_Patch = image_gt[stride * row:stride * (row + 1), stride * col:stride * (col + 1), :]
                        mask_Patch = mask[stride * row:stride * (row + 1), stride * col:stride * (col + 1)]

                        cv2.imwrite(image_save_path2 + name, image_Patch)
                        cv2.imwrite(image_gt_save_path2 + name, image_gt_Patch)
                        cv2.imwrite(mask_save_path2 + name, mask_Patch)
                        index += 1

                        f.write(image_save_path2 + name + "\n")
            f.close()


def re_crop(dataroot=None, stride=256, re_data_path=None,image_save_path=None):
    save_txt_path = r"E:\2021\dataset\earse\train_resize_patch_%s" % str(stride)
    image_list = ReadConfigTxt(dataroot, img_suffix=".jpg", save_path=save_txt_path)

    # generate path
    # re_data_path = r"E:\2021\dataset\earse\train_resize_%s\images" % str(stride)
    # image_save_path = r"E:\2021\dataset\earse\train_resize_%s\re_images" % str(stride)
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    for i, img_path in enumerate(tqdm(image_list["val"])):
        image = cv2.imread(img_path)

        # 判断分割策略；比例
        H, W = image.shape[:2]
        mode = cut_mode(H, W)
        new_W = mode[1] * stride
        new_H = mode[0] * stride

        # image_re
        image_re = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_AREA)

        index = 0
        for col in range(mode[1]):
            for row in range(mode[0]):
                name = img_path.split("images")[-1].replace(".", "_" + str(index) + ".")
                image_Patch = cv2.imread(re_data_path + name)

                image_re[stride * row:stride * (row + 1), stride * col:stride * (col + 1), :] = image_Patch

                index += 1
        image_re = cv2.resize(image_re, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_save_path + img_path.split("images")[-1], image_re)


# 根据划分比例 生成数据的txt
def GenConfigTxt(data_root, img_suffix=".bmp", split_str="||", train_test_ratio=1.0, save_path=None):
    """
    :param data_root:  数据集路径
    :param img_suffix: 图像的后缀
    :param split_str: 路径　标签之间拼接的分割符号
    :param train_test_ratio: 训练和测试的比例
    :return:
    """
    images_path = glob(os.path.join(data_root, '*{}'.format(img_suffix)))

    # random.shuffle(images_path)

    # 划分点
    seg_point = int(len(images_path) * train_test_ratio)
    train_list = images_path[0:int(0.8 * seg_point)]
    test_list = images_path[seg_point:len(images_path)]
    val_list = images_path[int(0.8 * seg_point):int(seg_point)]

    phases = ["train", "val", "test"]
    image_list = {"train": train_list, "val": val_list, "test": test_list}
    for phase in phases:
        # path=os.path.join(data_root, "{}.txt".format(phase))
        with open(os.path.join(save_path, "{}.txt".format(phase)), mode='w') as f:
            for i in range(len(image_list[phase])):
                image_path = image_list[phase][i]

                f.write(image_path + "\n")
        f.close()


def ReadConfigTxt(data_root, update_txt=False, img_suffix=".bmp", split_str="||", train_test_ratio=1.0, save_path=None):
    """
    :param data_root:  数据集路径
    :param img_suffix: 图像的后缀
    :param split_str: 路径　标签之间拼接的分割符号
    :param train_test_ratio: 训练和测试的比例
    :param update_txt: 是否重新生成txt
    :return: image_list{"train":train_list, "val":val_list, "test":test_list}
    """
    phases = ["train", "val", "test"]
    image_list = {}

    # 不存在txt或者指定更新txt时执行
    if not os.path.exists(os.path.join(save_path, "{}.txt".format(phases[0]))) or update_txt:
        GenConfigTxt(data_root, img_suffix=img_suffix, split_str=split_str, train_test_ratio=train_test_ratio,
                     save_path=save_path
                     )

    for phase in phases:
        with open(os.path.join(save_path, "{}.txt".format(phase)), mode='r') as f:
            images = f.readlines()
        images = [img.strip() for img in images]
        image_list[phase] = images
    return image_list

def post():
    image_paths = glob(r"F:\hw\ori" + r"\*.jpg")
    for i, image_path in enumerate(image_paths):
        image_ori = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image_ori_gray = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)
        print(image_path)
        cv2.imshow("ori",image_ori)
        image_gen_path=image_path.replace("ori","gen")
        image_gen=cv2.imdecode(np.fromfile(image_gen_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image_gen_gray = cv2.cvtColor(image_gen, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gen", image_gen)
        image_sub = cv2.subtract(image_gen_gray, image_ori_gray)
        retval, mask = cv2.threshold(image_sub, 25, 1, cv2.THRESH_BINARY)
        retval1, mask1 = cv2.threshold(image_sub, 25, 255, cv2.THRESH_BINARY)
        cv2.imshow("mask",mask1)
        img_result=np.zeros_like(image_ori)

        img_result[:,:,0]=image_gen[:,:,0]*mask+image_ori[:,:,0]*(1-mask)
        img_result[:, :, 1] = image_gen[:, :, 1] * mask + image_ori[:, :, 1] * (1 - mask)
        img_result[:, :, 2] = image_gen[:, :, 2] * mask + image_ori[:, :, 2] * (1 - mask)

        cv2.imshow("result",img_result)
        cv2.waitKey()

if __name__ == '__main__':
    data_path = r"E:\2021\dataset\earse\dehw_train_dataset\images"
    stride = 256  # 剪切patch大小

    # 剪切图片,保存 txt 文件在 save_txt_path
    save_txt_path = r"E:\2021\dataset\earse\train_resize_patch_%s" % str(stride)  # txt保存路径
    crop(dataroot=data_path, stride=stride,save_txt_path=save_txt_path)

    # # 复原图片
    # re_data_path = r"E:\2021\dataset\earse\train_resize_patch_%s\images" % str(stride)  # 待修复图
    # image_save_path = re_data_path.replace('images','re_images')  # 保存路径
    # re_crop(dataroot=data_path, re_data_path=re_data_path, image_save_path=image_save_path, stride=stride)

    # # 复原图片
    # re_data_path = r"E:\2021\dataset\earse\train_resize_patch_%s\images" % str(stride)  # 待修复图
    # image_save_path = re_data_path.replace('images','re_images')  # 保存路径
    # re_crop(dataroot=data_path, re_data_path=re_data_path, image_save_path=image_save_path, stride=stride)
