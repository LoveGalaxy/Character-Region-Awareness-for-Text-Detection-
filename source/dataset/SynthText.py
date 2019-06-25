import os
import cv2
import random
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import transform as TR
from torchvision import transforms
try:
    import datautils
    import transutils
except:
    from . import datautils
    from . import transutils

class SynthText(Dataset):
    def __init__(self, data_dir_path=None, random_rote_rate=None, data_file_name=None, istrain=True, image_size=(3, 640, 640), down_rate=2, transform=None):
        # check data path
        if data_dir_path == None:
            data_dir_path = "./data/SynthText/SynthText/"
        if data_file_name == None:
            data_file_name = "gt.mat"
        self.data_dir_path = data_dir_path
        self.data_file_name = data_file_name
        print("load data, please wait a moment...")

        self.agt = scio.loadmat(os.path.join(self.data_dir_path, self.data_file_name))
        self.istrain = istrain
        self.gt = {}
        if istrain:
            self.gt["txt"] = self.agt["txt"][0][:-1][:-10000]
            self.gt["imnames"] = self.agt["imnames"][0][:-10000]
            self.gt["charBB"] = self.agt["charBB"][0][:-10000]
            self.gt["wordBB"] = self.agt["wordBB"][0][:-10000]
        else:
            self.gt["txt"] = self.agt["txt"][0][-10000:]
            self.gt["imnames"] = self.agt["imnames"][0][-10000:]
            self.gt["charBB"] = self.agt["charBB"][0][-10000:]
            self.gt["wordBB"] = self.agt["wordBB"][0][-10000:]

        self.image_size = image_size
        self.down_rate = down_rate
        self.transform = transform
        self.random_rote_rate = random_rote_rate
    
    def __len__(self):
        return self.gt["txt"].shape[0]
    
    def resize(self, image, char_label, word_laebl):
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
            char_label = char_label * (resize_h / h)
            word_laebl = word_laebl * (resize_h / h)
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
            char_label = char_label * (resize_w / w)
            word_laebl = word_laebl * (resize_w / w)
        return img, char_label, word_laebl

    def __getitem__(self, idx):
        img_name = self.gt["imnames"][idx][0]
        image  = Image.open(os.path.join(self.data_dir_path, img_name))
        char_label = self.gt["charBB"][idx].transpose(2, 1, 0)
        if len(self.gt["wordBB"][idx].shape) == 3:
            word_laebl = self.gt["wordBB"][idx].transpose(2, 1, 0)
        else:
            word_laebl = self.gt["wordBB"][idx].transpose(1, 0)[np.newaxis, :]
        txt_label = self.gt["txt"][idx]

        img, char_label, word_laebl = self.resize(image, char_label, word_laebl)

        if self.random_rote_rate:
            angel = random.randint(0-self.random_rote_rate, self.random_rote_rate)
            img, M = datautils.rotate(angel, img)

        # char_label = char_label / self.down_rate
        # word_laebl = word_laebl / self.down_rate

        # char_gt = np.zeros((int(self.image_size[1]/self.down_rate), int(self.image_size[2]/self.down_rate)))
        # aff_gt = np.zeros((int(self.image_size[1]/self.down_rate), int(self.image_size[2]/self.down_rate)))

        char_gt = np.zeros((int(self.image_size[1]), int(self.image_size[2])))
        aff_gt = np.zeros((int(self.image_size[1]), int(self.image_size[2])))

        
        line_boxes = []
        char_index = 0
        word_index = 0
        for txt in txt_label:
            for strings in txt.split("\n"):
                for string in strings.split(" "):
                    if string == "":
                        continue
                    char_boxes = []
                    for char in string:
                        x0, y0 = char_label[char_index][0]
                        x1, y1 = char_label[char_index][1]
                        x2, y2 = char_label[char_index][2]
                        x3, y3 = char_label[char_index][3]
                        
                        if self.random_rote_rate:
                            x0, y0 = datautils.rotate_point(M, x0, y0)
                            x1, y1 = datautils.rotate_point(M, x1, y1)
                            x2, y2 = datautils.rotate_point(M, x2, y2)
                            x3, y3 = datautils.rotate_point(M, x3, y3)
                        
                        x0, y0, x1, y1, x2, y2, x3, y3 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), int(round(x3)), int(round(y3))
                        char_boxes.append([x0, y0, x1, y1, x2, y2, x3, y3])
                        box, deta_x, deta_y = datautils.find_min_rectangle([x0, y0, x1, y1, x2, y2, x3, y3])
                        if deta_x <= 0 or deta_x >= self.image_size[2] or deta_y <= 0 or deta_y >= self.image_size[1]:
                            # print(idx, deta_x, deta_y)
                            char_index += 1
                            continue
                        try:
                            gaussian = datautils.gaussian_kernel_2d_opencv(kernel_size=(deta_y, deta_x))
                            pts = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                            res = datautils.aff_gaussian(gaussian, box, pts, deta_y, deta_x)
                        except:
                            char_index += 1
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
                            
                        char_index += 1
                    word_index += 1
                    line_boxes.append(char_boxes)
        affine_boxes = []
        for char_boxes in line_boxes:
            affine_boxes.extend(datautils.create_affine_boxes(char_boxes))
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
            # 'affine_boxes': affine_boxes,
            # 'line_boxes': line_boxes,
            # 'char_label': char_label
        }

        if self.transform:
            sample = self.transform(sample)

        sample['char_gt'] = TR.resize(sample['char_gt'], (int(self.image_size[1]/self.down_rate), int(self.image_size[2]/self.down_rate)))
        sample['aff_gt'] = TR.resize(sample['aff_gt'], (int(self.image_size[1]/self.down_rate), int(self.image_size[2]/self.down_rate)))

        return sample
        
if __name__ == "__main__":
    dataset = SynthText(transform=transforms.Compose([
                                                    transutils.RandomCrop((400, 400)),
                                                    transutils.Rescale((640, 640))
    ]))
    print(len(dataset))
    # idataset = iter(dataset)
    # data = next(idataset)
    dataLoader = DataLoader(dataset, 100, shuffle=False, num_workers=5)
    max_item = 0
    for bacth_data in dataLoader:
        print(max_item, bacth_data["image"].shape)
        print(max_item, bacth_data["char_gt"].shape)
        print(max_item, bacth_data["aff_gt"].shape)
        max_item += 1
        # images, confidence, local, classification = bacth_data['image'], bacth_data['confidence'], bacth_data["local"], bacth_data['classification']
        # print(images.shape, confidence.shape, local.shape, classification.shape)
        # # confidence = confidence>0
        # print(confidence[confidence==1].shape)
        



