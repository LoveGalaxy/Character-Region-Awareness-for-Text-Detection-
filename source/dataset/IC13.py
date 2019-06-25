import os
import cv2
import json
import random
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from skimage import transform as TR
try:
    import datautils
    import transutils
except:
    from . import datautils
    from . import transutils


class IC13(Dataset):
    def __init__(self, data_dir_path="./data/ICDAR2013/", image_size=(3, 640, 640), isTrain=True, random_rote_rate=None, transform=None, down_rate=2):
        self.data_dir = data_dir_path
        self.image_size = image_size
        self.down_rate = down_rate
        self.isTrain = isTrain
        if isTrain:
            img_path = "./train/Challenge2_Training_Task12_Images/"
            gt_path = "./train/Challenge2_Training_Task2_GT/"
            self.img_dir = os.path.join(data_dir_path, img_path)
            self.gt_dir = os.path.join(data_dir_path, gt_path)
            self.images = os.listdir(self.img_dir)
        else:
            img_path = "./test/Challenge2_Test_Task12_Images/"
            gt_path = "./test/Challenge2_Test_Task2_GT/"
            self.img_dir = os.path.join(data_dir_path, img_path)
            self.gt_dir = os.path.join(data_dir_path, gt_path)
            self.images = os.listdir(self.img_dir)
        self.transform = transform
        self.random_rote_rate = random_rote_rate

    def __len__(self):
        return len(self.images)
    
    def resize(self, image, line_boxes):
        w, h = image.size
        img = np.zeros(self.image_size)
        rate = self.image_size[2] / self.image_size[1]
        rate_pic = w / h
        
        if rate_pic > rate:
            resize_h = int(self.image_size[2] / rate_pic)
            image = image.resize((self.image_size[2], resize_h), Image.ANTIALIAS)
            image = np.array(image)
            if self.image_size[0] == 3:
                if len(image.shape) == 2:
                    image = np.tile(image, (3, 1, 1))
                else:
                    image = image.transpose((2, 0, 1))
            img[:,:resize_h,:] = image
            for i in range(len(line_boxes)):
                for j in range(len(line_boxes[i])):
                    for k in range(len(line_boxes[i][j])):
                        line_boxes[i][j][k] = line_boxes[i][j][k] * (resize_h / h) #/ self.down_rate
        else:
            resize_w = int(rate_pic * self.image_size[1])
            image = image.resize((resize_w, self.image_size[1]), Image.ANTIALIAS)
            image = np.array(image)
            if self.image_size[0] == 3:
                if len(image.shape) == 2:
                    image = np.tile(image, (3, 1, 1))
                else:
                    image = image.transpose((2, 0, 1))

            img[:,:,:resize_w] = np.array(image)
            for i in range(len(line_boxes)):
                for j in range(len(line_boxes[i])):
                    for k in range(len(line_boxes[i][j])):
                        line_boxes[i][j][k] = line_boxes[i][j][k] * (resize_w / w) #/ self.down_rate
        return img, line_boxes


    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.images[idx])
        if self.isTrain:
            gt_path = os.path.join(self.gt_dir, self.images[idx].split(".")[0] + "_GT.txt")
        else:
            gt_path = os.path.join(self.gt_dir, "gt_" + self.images[idx].split(".")[0] + ".txt")
        image = Image.open(image_path)
        with open(gt_path, "r") as f:
            lines = f.readlines()
        
        line_boxes = []
        line_box = []
        for line in lines:
            if line == '\n':
                line_boxes.append(line_box)
                line_box = []
                continue
            data = line.split(" ")
            x0, y0, x3, y3 = data[5:9]
            x0, y0, x3, y3 = int(x0), int(y0), int(x3), int(y3)
            line_box.append([x0, y0, x3, y0, x3, y3, x0, y3])
        line_boxes.append(line_box)
        
        img, line_boxes = self.resize(image, line_boxes)
        if self.random_rote_rate:
            angel = random.randint(0-self.random_rote_rate, self.random_rote_rate)
            img, M = datautils.rotate(angel, img)
        # char_gt = np.zeros((int(self.image_size[1]/self.down_rate), int(self.image_size[2]/self.down_rate)))
        # aff_gt = np.zeros((int(self.image_size[1]/self.down_rate), int(self.image_size[2]/self.down_rate)))
        char_gt = np.zeros((int(self.image_size[1]), int(self.image_size[2])))
        aff_gt = np.zeros((int(self.image_size[1]), int(self.image_size[2])))

        
        for line_box in line_boxes:
            for box in line_box:
                x0, y0, x1, y1, x2, y2, x3, y3 = box
                if self.random_rote_rate:
                    x0, y0 = datautils.rotate_point(M, x0, y0)
                    x1, y1 = datautils.rotate_point(M, x1, y1)
                    x2, y2 = datautils.rotate_point(M, x2, y2)
                    x3, y3 = datautils.rotate_point(M, x3, y3)
                x0, y0, x1, y1, x2, y2, x3, y3 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), int(round(x3)), int(round(y3))
                minbox, deta_x, deta_y = datautils.find_min_rectangle([x0, y0, x1, y1, x2, y2, x3, y3])
                if deta_x <= 0 or deta_x >= self.image_size[2] or deta_y <= 0 or deta_y >= self.image_size[1]:
                    continue
                try:
                    gaussian = datautils.gaussian_kernel_2d_opencv(kernel_size=(deta_y, deta_x))
                    pts = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                    res = datautils.aff_gaussian(gaussian, minbox, pts, deta_y, deta_x)
                except:
                    continue
                min_x = min(x0, x1, x2, x3)
                min_y = min(y0, y1, y2, y3)

                if np.max(res) > 0:
                    mx = 1 / np.max(res)
                    res = mx * res
                    gh, gw = res.shape
                    for th in range(gh):
                        for tw in range(gw):
                            if 0 < min_y+th < char_gt.shape[0] and 0 < min_x+tw < char_gt.shape[1]:
                                try:
                                    char_gt[min_y+th, min_x+tw] = max(char_gt[min_y+th, min_x+tw], res[th, tw])
                                except:
                                    print(idx, min_y+th, min_x+tw)
            affine_boxes = datautils.create_affine_boxes(line_box)
            for points in affine_boxes:
                x0, y0, x1, y1, x2, y2, x3, y3 = points[0], points[1], points[2], points[3], points[4], points[5], points[6], points[7]
                box, deta_x, deta_y = datautils.find_min_rectangle(points)
                if deta_x <= 0 or deta_x >= self.image_size[2] or deta_y <= 0 or deta_y >= self.image_size[1]:
                    continue
                try:
                    gaussian = datautils.gaussian_kernel_2d_opencv(kernel_size=(deta_y, deta_x))
                    pts = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                    res = datautils.aff_gaussian(gaussian, box, pts,  deta_y, deta_x)
                except:
                    continue
                min_x = min(x0, x1, x2, x3)
                min_y = min(y0, y1, y2, y3)

                if np.max(res) > 0:
                    mx = 1 / np.max(res)
                    res = mx * res
                    gh, gw = res.shape
                    for th in range(gh):
                        for tw in range(gw):
                            if 0 < min_y+th < aff_gt.shape[0] and 0 < min_x+tw < aff_gt.shape[1]:
                                try:
                                    aff_gt[min_y+th, min_x+tw] = max(aff_gt[min_y+th, min_x+tw], res[th, tw])
                                except:
                                    print(idx, min_y+th, min_x+tw)       


        sample = {
            'image': img,
            'char_gt': char_gt,
            'aff_gt': aff_gt,
        }
        if self.transform:
            sample = self.transform(sample)

        sample['char_gt'] = TR.resize(sample['char_gt'], (int(self.image_size[1]/self.down_rate), int(self.image_size[2]/self.down_rate)))
        sample['aff_gt'] = TR.resize(sample['aff_gt'], (int(self.image_size[1]/self.down_rate), int(self.image_size[2]/self.down_rate)))
        return sample

if __name__ == "__main__":
    dataset = IC13()
    print(len(dataset))

    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=0)
    # idata = iter(dataset)
    # # next(idata)
    # # next(idata)
    for batch_data in dataloader:
        print(batch_data["image"].shape, batch_data["char_gt"].shape, batch_data["aff_gt"].shape)