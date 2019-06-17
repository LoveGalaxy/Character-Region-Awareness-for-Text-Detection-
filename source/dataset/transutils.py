import numpy as np 
from torchvision import transforms, utils
from skimage import io, transform
import random
import cv2

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    def __call__(self, sample):
        image, char_gt, aff_gt = sample["image"], sample["char_gt"], sample["aff_gt"]
        h, w = image.shape[1:]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top:top+new_h, left:left+new_w]
        char_gt = char_gt[top:top+new_h, left:left+new_w]
        aff_gt = aff_gt[top:top+new_h, left:left+new_w]
        
        sample = {'image': image, 'char_gt': char_gt, 'aff_gt': aff_gt}
        return sample

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, sample):
        image, char_gt, aff_gt = sample["image"], sample["char_gt"], sample["aff_gt"]
        h, w = image.shape[1:]
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        image = image.transpose((1, 2, 0))
        image = transform.resize(image, (new_h, new_w))
        char_gt = transform.resize(char_gt, (new_h, new_w))
        aff_gt = transform.resize(aff_gt, (new_h, new_w))
        image = image.transpose((2, 0, 1))
        sample = {'image': image, 'char_gt': char_gt, 'aff_gt': aff_gt}
        return sample

class RedomRescale(object):
    def __init__(self, output_size_list):
        self.output_size_list
    
    def __call__(self, sample):
        length = len(self.output_size_list)
        idx = random.randint(0, length-1)
        


        return sample

class Random_change(object):
    def __init__(self, random_bright, random_swap, random_contrast, random_saturation, random_hue):
        self.random_bright = random_bright
        self.random_swap = random_swap
        self.random_contrast = random_contrast
        self.random_saturation = random_saturation
        self.random_hue = random_hue

    
    def __call__(self, sample):
        image, char_gt, aff_gt = sample["image"], sample["char_gt"], sample["aff_gt"]
        # 亮度
        if random.random() < self.random_bright:
            delta = random.uniform(-32, 32)
            image += delta
            image = image.clip(min=0, max=255)

        # 变换通道
        perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
        if random.random() < self.random_swap:
            swap = perms[random.randrange(0, len(perms))]
            image = image[swap, :, :]
        
        # 对比度
        if random.random() < self.random_contrast:
            alpha = random.uniform(0.5, 1.5)
            image *= alpha
            image = image.clip(min=0, max=255)

        # 饱和度
        if random.random() < self.random_saturation:
            image[1, :, :] *= random.uniform(0.5, 1.5)
        
        # 色度
        if random.random() < self.random_hue:
            image[0, :, :] += random.uniform(-18.0, 18.0)
            image[0, :, :][image[0, :, :] > 360.0] -= 360.0
            image[0, :, :][image[0, :, :] < 0.0] += 360.0

        sample = {'image': image, 'char_gt': char_gt, 'aff_gt': aff_gt}
        return sample
    



if __name__ == "__main__":
    
    img = np.zeros((3, 300, 300))
    point = (150, 150, 1)
    r = Rotate(30)
    image = r(img, point)
