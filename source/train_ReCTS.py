import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from luo import TicToc
from torchvision import transforms
from models import network
from models import ohem

from dataset import SynthText
from dataset import ReCTS
from dataset import transutils

def run():
    print(TicToc.format_time(), "begining.......")
    image_size = (3, 640, 640)
    train_dataset = ReCTS.ReCTS(image_size=image_size,
                        random_rote_rate=30,
                        data_dir_path="./data/ReCTS/", 
                       transform=transforms.Compose([
                                    transutils.RandomCrop((480, 480)),
                                    transutils.Rescale((640, 640)),
                                    transutils.Random_change(0.5, 0.5, 0.5, 0.5, 0.5)]))
    train_dataLoader = DataLoader(train_dataset, 4, shuffle=True, num_workers=4)
    test_dataset = ReCTS.ReCTS(image_size=image_size, istrain=False)
    test_dataLoader = DataLoader(test_dataset, 4, shuffle=False, num_workers=4)

    vgg = network.UP_VGG()
    vgg = vgg.to("cuda")
    vgg = torch.load("./model/vgg_Sy_0.pkl")
    print(TicToc.format_time(), "finish load")


    optimizer = torch.optim.Adam(vgg.parameters(), lr = 0.001)
    # crite = nn.MSELoss(reduction="mean")
    # l1_crite = nn.SmoothL1Loss(reduction="mean")
    # cls_crite = nn.CrossEntropyLoss(reduction="mean")
    loss_fn = ohem.MSE_OHEM_Loss()
    loss_fn = loss_fn.to("cuda")
    print(TicToc.format_time(), " training.........")

    for e in range(100):
        # train
        total_loss = 0.0
        char_loss = 0.0
        aff_loss = 0.0
        b_total_loss = 0.0
        b_char_loss = 0.0
        b_aff_loss = 0.0
        vgg.train()

        for i, batch_data in enumerate(train_dataLoader):
            image = batch_data["image"].type(torch.FloatTensor) / 255 - 0.5
            image = image.to("cuda")
            reg, aff = vgg(image)
            predict_r = torch.squeeze(reg, dim=1)
            predict_l = torch.squeeze(aff, dim=1)

            targets_r = batch_data["char_gt"].type(torch.FloatTensor)
            targets_r = targets_r.to("cuda")

            targets_l = batch_data["aff_gt"].type(torch.FloatTensor)
            targets_l = targets_l.to("cuda")

            optimizer.zero_grad()
            loss_r = loss_fn(predict_r, targets_r)
            loss_l = loss_fn(predict_l, targets_l)

            loss = loss_r + loss_l #+ loss_cls
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            char_loss += loss_r.item()
            aff_loss += loss_l.item()
            b_total_loss += loss.item()
            b_char_loss += loss_r.item()
            b_aff_loss += loss_l.item()

            if i % 1000 == 999:
                print(TicToc.format_time(), e, i, b_total_loss, b_char_loss, b_aff_loss)
                b_total_loss = 0.0
                b_char_loss = 0.0
                b_aff_loss = 0.0
        print("Train ", TicToc.format_time(), e, total_loss, char_loss, aff_loss)

        # test
        min_test_loss = 100000
        with torch.no_grad():
            test_total_loss = 0.0
            test_char_loss = 0.0
            test_aff_loss = 0.0
            
            for i, batch_data in enumerate(test_dataLoader):
                image = batch_data["image"].type(torch.FloatTensor) / 255 - 0.5
                image = image.to("cuda")
                reg, aff = vgg(image)
                predict_r = torch.squeeze(reg, dim=1)
                predict_l = torch.squeeze(aff, dim=1)

                targets_r = batch_data["char_gt"].type(torch.FloatTensor)
                targets_r = targets_r.to("cuda")

                targets_l = batch_data["aff_gt"].type(torch.FloatTensor)
                targets_l = targets_l.to("cuda")

                loss_r = loss_fn(predict_r, targets_r)
                loss_l = loss_fn(predict_l, targets_l)

                loss = loss_r + loss_l 

                test_total_loss += loss.item()
                test_char_loss += loss_r.item()
                test_aff_loss += loss_l.item()
            print("Test ", TicToc.format_time(), e, i, test_total_loss, test_char_loss, test_aff_loss)
        if min_test_loss > test_total_loss:
            min_test_loss = test_total_loss
            print("save: {0}".format(e))
            torch.save(vgg, "./model/vgg_ReCTS_{0}.pkl".format(e))

if __name__ == "__main__":
    run()

