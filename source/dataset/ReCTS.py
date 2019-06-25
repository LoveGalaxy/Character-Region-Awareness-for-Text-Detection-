import os
import cv2
import json
import random
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import transform as TR

try:
    import datautils
except:
    from . import datautils


class ReCTS(Dataset):
    def __init__(self, data_dir_path="./data/ReCTS/", dir_list=[1, 2, 3, 4], random_rote_rate=None, isTrain=True, image_size=(3, 640, 640), down_rate=2, transform=None):
        self.sub_dirs = ["ReCTS_part{0}".format(i) for i in dir_list]
        self.image_size = image_size
        image_list = []
        label_list = []
        for sub_dir in self.sub_dirs:
            tmp_label_list = []
            tmp_label_list.extend([label if label[:2] != "._" else label[2:] for label in os.listdir(os.path.join(data_dir_path, sub_dir, "gt"))])
            image_list.extend([os.path.join(data_dir_path, sub_dir, "img", label[:-5]+".jpg") for label in tmp_label_list])
            label_list.extend([os.path.join(data_dir_path, sub_dir, "gt", label) for label in tmp_label_list])
        if isTrain:
            self.label_list = label_list[:30000]
            self.image_list = image_list[:30000]
        else:
            self.label_list = label_list[30000:]
            self.image_list = image_list[30000:]
        self.down_rate = down_rate
        self.transform = transform
        self.random_rote_rate = random_rote_rate
        # self.char_to_id, self.id_to_char self.num_classes = self._get_dic(dic_file_path)

    def __len__(self):
        return len(self.label_list)

    def box_in(self, line, char, thr=0.95):
        lx0, ly0, lx1, ly1, lx2, ly2, lx3, ly3 = line
        cx0, cy0, cx1, cy1, cx2, cy2, cx3, cy3 = char
        
        l_lt_x = min(lx0, lx1, lx2, lx3)
        l_lt_y = min(ly0, ly1, ly2, ly3)
        l_rd_x = max(lx0, lx1, lx2, lx3)
        l_rd_y = max(ly0, ly1, ly2, ly3)

        c_lt_x = min(cx0, cx1, cx2, cx3)
        c_lt_y = min(cy0, cy1, cy2, cy3)
        c_rd_x = max(cx0, cx1, cx2, cx3)
        c_rd_y = max(cy0, cy1, cy2, cy3)

        S_c = (c_rd_x-c_lt_x) * (c_rd_y - c_lt_y)

        if S_c <= 0:
            return False
        overlap_X = (l_rd_x-l_lt_x) + (c_rd_x-c_lt_x) - (max(l_rd_x, c_rd_x) - min(l_lt_x, c_lt_x))
        overlap_Y = (l_rd_y-l_lt_y) + (c_rd_y-c_lt_y) - (max(l_rd_y, c_rd_y) - min(l_lt_y, c_lt_y))
        if overlap_X <= 0 or overlap_Y <= 0:
            return False
        S_U = overlap_X * overlap_Y
        
        if 1 >= S_U / S_c > thr:
            #print(S_U, S_c, S_U / S_c)
            return True
        # print(line, char)
        # print(S_U, S_c, S_U / S_c)
        return False
    
    def resize(self, image, lines_loc, line_boxes):
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
            lines_loc = lines_loc * (resize_h / h) #/ self.down_rate
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
            lines_loc = lines_loc * (resize_w / w) #/ self.down_rate
            for i in range(len(line_boxes)):
                for j in range(len(line_boxes[i])):
                    for k in range(len(line_boxes[i][j])):
                        line_boxes[i][j][k] = line_boxes[i][j][k] * (resize_w / w) #/ self.down_rate
        return img, lines_loc, line_boxes

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        label_path = self.label_list[idx]
        image = Image.open(image_path)
        # print(label_path)

        with open(label_path, "r") as f:
            label_json = json.load(f)

        lines_loc = []
        chars_loc = []
        line_boxes = []
        char_num = 0
        total_char_num = len(label_json["chars"])
        for line in label_json["lines"]:
            label = line["transcription"]
            if len(label) == 1:
                line_boxes.append([line["points"]])
            if label == "###":
                continue
            if label != "###":
                lines_loc.append(line["points"])
            line_box = []
            for c in label:
                
                while char_num < total_char_num:
                    c_label = label_json["chars"][char_num]["transcription"]
                    if c_label == c:
                        if label_json["chars"][char_num]["points"] != [-1, -1, -1, -1, -1, -1, -1, -1]:
                            line_box.append(label_json["chars"][char_num]["points"])
                        char_num += 1
                        break
                    char_num += 1
            # print(label, line_box)
            line_boxes.append(line_box)

        
        # for char in label_json["chars"]:
        #     label = char["transcription"]
        #     if label != "###" and char["points"] != [-1, -1, -1, -1, -1, -1, -1, -1]:
        #         chars_loc.append(char["points"])

        # line_boxes = []
        # for line in lines_loc:
        #     line_box = []
        #     for char in chars_loc:
        #         if self.box_in(line, char):
        #             line_box.append(char)
        #     line_boxes.append(line_box)

        xlines_loc = []
        xline_boxes = []
        for loc, box in zip(lines_loc, line_boxes):
            if box != []:
                xlines_loc.append(loc)
                xline_boxes.append(box)

        lines_loc = np.array(xlines_loc)
        img, lines_loc, line_boxes = self.resize(image, lines_loc, line_boxes)
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
            

        # print(idx, w, h)
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
    dataset = ReCTS()
    print(len(dataset))

    dataloader = DataLoader(dataset, 100, shuffle=True, num_workers=0)
    # idata = iter(dataset)
    # # next(idata)
    # # next(idata)
    for batch_data in dataloader:
        print(batch_data["image"].shape)