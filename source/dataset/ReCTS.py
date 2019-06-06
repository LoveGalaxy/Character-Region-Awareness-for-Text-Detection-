import os
import cv2
import json
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import datautils
except:
    from . import datautils


class ReCTS(Dataset):
    def __init__(self, data_dir="./data/ReCTS/", dir_list=[1, 2, 3, 4], istrain=True, image_size=(3, 640, 640), down_rate=2):
        self.sub_dirs = ["ReCTS_part{0}".format(i) for i in dir_list]
        self.image_size = image_size
        image_list = []
        label_list = []
        for sub_dir in self.sub_dirs:
            tmp_label_list = []
            tmp_label_list.extend([label if label[:2] != "._" else label[2:] for label in os.listdir(os.path.join(data_dir, sub_dir, "gt"))])
            image_list.extend([os.path.join(data_dir, sub_dir, "img", label[:-5]+".jpg") for label in tmp_label_list])
            label_list.extend([os.path.join(data_dir, sub_dir, "gt", label) for label in tmp_label_list])
        if istrain:
            self.label_list = label_list[:30000]
            self.image_list = image_list[:30000]
        else:
            self.label_list = label_list[30000:]
            self.image_list = image_list[30000:]
        self.down_rate = down_rate
        # self.char_to_id, self.id_to_char self.num_classes = self._get_dic(dic_file_path)

    def __len__(self):
        return len(self.label_list)

    def box_in(self, line, char, thr=0.5):
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
            lines_loc = lines_loc * (resize_h / h)
            for i in range(len(line_boxes)):
                for j in range(len(line_boxes[i])):
                    for k in range(len(line_boxes[i][j])):
                        line_boxes[i][j][k] = line_boxes[i][j][k] * (resize_h / h)
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
            lines_loc = lines_loc * (resize_w / w)
            for i in range(len(line_boxes)):
                for j in range(len(line_boxes[i])):
                    for k in range(len(line_boxes[i][j])):
                        line_boxes[i][j][k] = line_boxes[i][j][k] * (resize_w / w)
        return img, lines_loc, line_boxes

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        label_path = self.label_list[idx]
        image = Image.open(image_path)

        with open(label_path, "r") as f:
            label_json = json.load(f)

        lines_loc = []
        chars_loc = []

        for line in label_json["lines"]:
            label = line["transcription"]
            if len(label) == 1:
                chars_loc.append(line["points"])
            if label != "###":
                lines_loc.append(line["points"])
        
        for char in label_json["chars"]:
            label = char["transcription"]
            if label != "###" and char["points"] != [-1, -1, -1, -1, -1, -1, -1, -1]:
                chars_loc.append(char["points"])
        line_boxes = []
        for line in lines_loc:
            line_box = []
            for char in chars_loc:
                if self.box_in(line, char):
                    line_box.append(char)
            line_boxes.append(line_box)

        xlines_loc = []
        xline_boxes = []
        for loc, box in zip(lines_loc, line_boxes):
            if box != []:
                xlines_loc.append(loc)
                xline_boxes.append(box)

        lines_loc = np.array(xlines_loc)
        # line_boxes = np.array(xline_boxes)
        # print(len(xlines_loc), len(xline_boxes), lines_loc.shape, len(line_boxes))
        # print(char_loc)
        img, lines_loc, line_boxes = self.resize(image, lines_loc, line_boxes)
        img = np.zeros(self.image_size)

        # print(idx, w, h)
        sample = {
            'image': img,
        }
        return sample

if __name__ == "__main__":
    dataset = ReCTS()
    print(len(dataset))

    dataloader = DataLoader(dataset, 100, shuffle=True, num_workers=0)
    # idata = iter(dataset)
    # next(idata)
    # next(idata)
    for batch_data in dataloader:
        print(batch_data["image"].shape)