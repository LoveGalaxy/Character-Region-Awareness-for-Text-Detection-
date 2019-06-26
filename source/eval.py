import os
import cv2
from PIL import Image
import numpy as np
import torch

def load_imgs_dir(dir_path):
    images = os.listdir(dir_path)
    r_images = []
    for image in images:
        r_images.append(os.path.join(dir_path, image))
    return r_images

def read_image(img_path, image_size=(3, 320, 320)):
    image = Image.open(img_path)
    w, h = image.size
    img = np.zeros(image_size)
    rate = image_size[2] / image_size[1]
    rate_pic = w / h
    
    if rate_pic > rate:
        resize_h = int(image_size[2] / rate_pic)
        image = image.resize((image_size[2], resize_h), Image.ANTIALIAS)
        image = np.array(image)
        if image_size[0] == 3:
            if len(image.shape) == 2:
                image = np.tile(image, (3, 1, 1))
            else:
                image = image.transpose((2, 0, 1))
        img[:,:resize_h,:] = image
        resize_rate = h / resize_h
    else:
        resize_w = int(rate_pic * image_size[1])
        image = image.resize((resize_w, image_size[1]), Image.ANTIALIAS)
        image = np.array(image)
        if image_size[0] == 3:
            if len(image.shape) == 2:
                image = np.tile(image, (3, 1, 1))
            else:
                image = image.transpose((2, 0, 1))
        img[:,:,:resize_w] = np.array(image)
        resize_rate = w / resize_w
    c, h, w = img.shape
    return img.reshape(1, c, h, w), resize_rate

def eval_IC13(model, write_dir="./out/IC13/", file_path="./data/IC13Test/", image_size=(3, 640, 640), thread_a=0.1, thread_c=0.1):
    images = os.listdir(file_path)
    
    for image_name in images:
        name = image_name.split(".")[0]
        image_path = os.path.join(file_path, image_name)
        img, rate = read_image(image_path, image_size)
        img = img / 255 - 0.5
        with torch.no_grad():
            img = torch.from_numpy(img).type(torch.FloatTensor).to("cuda")
            reg, aff = model(img)
            nreg = reg.to("cpu").detach().numpy()
        nreg = nreg.reshape(nreg.shape[2], nreg.shape[3])
        naff = aff.to("cpu").detach().numpy()
        naff = naff.reshape(naff.shape[2], naff.shape[3])
        m = (nreg > thread_c) + (naff > thread_a)
        m = m.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m, 4, cv2.CV_32S)
        data = ""
        for i in range(num_labels-1):
            if stats[i+1][-1] > 20: #and stats[i+1][-3] > 10:
                x0, y0, deta_x, deta_y, _ = stats[i+1] * 2 * rate
                data += "{0},{1},{2},{3}\n".format(int(x0), int(y0), int(x0+deta_x), int(y0+deta_y))
        with open(os.path.join(write_dir, "res_"+name+".txt"), "w") as f:
            f.writelines(data)
            

    

        