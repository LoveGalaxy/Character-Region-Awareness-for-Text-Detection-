import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from luo import TicToc

def run():
    image_size = (3, 320, 320)

    train_dataset = ReCTS(img_size=image_size, ischar=True)
    train_dataLoader = DataLoader(train_dataset, 64, shuffle=False, num_workers=4)
    test_dataset = ReCTS(img_size=image_size, ischar=True, istrain=False)
    test_dataLoader = DataLoader(test_dataset, 64, shuffle=False, num_workers=4)

    backbone = resnet.resnet18(pretrained=True)
    lol = CenterNet(backbone, train_dataset.num_classes).to("cuda")
    print(TicToc.format_time())
    print(lol)

    optimizer = torch.optim.Adam(lol.parameters(), lr = 0.01)
    crite = nn.MSELoss(reduction="mean")
    l1_crite = nn.SmoothL1Loss(reduction="mean")
    cls_crite = nn.CrossEntropyLoss(reduction="mean")

    iou = compute_iou()
    for e in range(100):
        # train
        total_loss = 0.0
        conf_loss = 0.0
        local_loss = 0.0
        class_loss = 0.0
        total_count = 0
        all_count = 0
        lol.train()

        for i, batch_data in enumerate(train_dataLoader):
            image = batch_data["image"].type(torch.FloatTensor) / 255 - 0.5
            image = image.to("cuda")
            confidence, local = lol(image)
            predict_c = torch.squeeze(confidence, dim=1)

            targets_c = batch_data["confidence"].type(torch.FloatTensor)
            targets_c = targets_c.to("cuda")

            targets_l = batch_data["local"].type(torch.FloatTensor)
            targets_l = targets_l.to("cuda")

            # targets_cls = batch_data["classification"].type(torch.LongTensor)
            # targets_cls = targets_cls.to("cuda")
            # confid = targets_c.unsqueeze(3).reshape(-1, 1)
            # con_mask = confid[:,0] > 0.9
            # targets_cls = targets_cls.reshape(-1)
            # clsfication = clsfication.permute(0, 2, 3, 1).reshape(-1, train_dataset.num_classes)

            optimizer.zero_grad()
            loss_c = crite(predict_c, targets_c)
            loss_l = l1_crite(local, targets_l)

            loss = loss_c + 0.1 * loss_l #+ loss_cls
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            conf_loss += loss_c.item()
            local_loss += loss_l.item()

            count, a_count = iou(targets_c, targets_l, predict_c, local)
            total_count += count
            all_count += a_count
        print(TicToc.format_time(), e, i, total_loss, conf_loss, local_loss, total_count, all_count)
        # test
        lol.eval()
        test_total_loss = 0.0
        test_conf_loss = 0.0
        test_local_loss = 0.0
        test_class_loss = 0.0
        test_total_count = 0
        test_all_count = 0
        
        for i, batch_data in enumerate(test_dataLoader):
            image = batch_data["image"].type(torch.FloatTensor) / 255 - 0.5
            image = image.to("cuda")
            confidence, local = lol(image)
            predict_c = torch.squeeze(confidence, dim=1)

            targets_c = batch_data["confidence"].type(torch.FloatTensor)
            targets_c = targets_c.to("cuda")

            targets_l = batch_data["local"].type(torch.FloatTensor)
            targets_l = targets_l.to("cuda")
            optimizer.zero_grad()
            loss_c = crite(predict_c, targets_c)
            loss_l = l1_crite(local, targets_l)

            loss = loss_c + 0.1 * loss_l #+ loss_cls
            loss.backward()
            optimizer.step()

            test_total_loss += loss.item()
            test_conf_loss += loss_c.item()
            test_local_loss += loss_l.item()

            count, a_count = iou(targets_c, targets_l, predict_c, local)
            test_total_count += count
            test_all_count += a_count
            #class_loss += loss_cls.item()
        print(TicToc.format_time(), e, i, test_total_loss, test_conf_loss, test_local_loss, test_total_count, test_all_count)#, class_loss)
        if e % 10 == 9:
            torch.save(lol, "./model/lol_model_2_{0}.pkl".format(e))

if __name__ == "__main__":
    run()

