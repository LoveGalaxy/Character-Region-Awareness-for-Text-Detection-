import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from luo import TicToc

from models import network
from dataset import SynthText

def run():
    image_size = (3, 320, 320)

    train_dataset = SynthText.SynthText(image_size=(3, 320, 320))
    train_dataLoader = DataLoader(train_dataset, 16, shuffle=True, num_workers=4)
    test_dataset = SynthText.SynthText(image_size=image_size, istrain=False)
    test_dataLoader = DataLoader(test_dataset, 64, shuffle=True, num_workers=4)

    vgg = network.UP_VGG()
    vgg = vgg.to("cuda")
    print(TicToc.format_time())


    optimizer = torch.optim.Adam(vgg.parameters(), lr = 0.01)
    crite = nn.MSELoss(reduction="mean")
    l1_crite = nn.SmoothL1Loss(reduction="mean")
    cls_crite = nn.CrossEntropyLoss(reduction="mean")

    for e in range(100):
        # train
        total_loss = 0.0
        char_loss = 0.0
        aff_loss = 0.0
        class_loss = 0.0
        total_count = 0
        all_count = 0
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

            # targets_cls = batch_data["classification"].type(torch.LongTensor)
            # targets_cls = targets_cls.to("cuda")
            # confid = targets_c.unsqueeze(3).reshape(-1, 1)
            # con_mask = confid[:,0] > 0.9
            # targets_cls = targets_cls.reshape(-1)
            # clsfication = clsfication.permute(0, 2, 3, 1).reshape(-1, train_dataset.num_classes)

            optimizer.zero_grad()
            loss_r = crite(predict_r, targets_r)
            loss_l = crite(predict_l, targets_l)

            loss = loss_r + loss_l #+ loss_cls
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            char_loss += loss_r.item()
            aff_loss += loss_l.item()
        print(TicToc.format_time(), e, i, total_loss, char_loss, aff_loss)

        # test
        with torch.no_grad():
            test_total_loss = 0.0
            test_char_loss = 0.0
            test_aff_loss = 0.0

            min_test_loss = 0.0
            
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

                loss_r = crite(predict_r, targets_r)
                loss_l = crite(predict_l, targets_l)

                loss = loss_r + loss_l 

                test_total_loss += loss.item()
                test_char_loss += loss_r.item()
                test_aff_loss += loss_l.item()
            print(TicToc.format_time(), e, i, test_total_loss, test_char_loss, test_aff_loss)

        if e == 0:
            min_test_loss = test_total_loss
            torch.save(vgg, "./model/vgg.pkl")
        elif total_loss < min_test_loss:
            print("save: ", e)
            min_test_loss = test_total_loss
            torch.save(vgg, "./model/vgg.pkl")

if __name__ == "__main__":
    run()

