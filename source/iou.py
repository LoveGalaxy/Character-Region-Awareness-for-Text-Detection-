import numpy as np 
import cv2

def iou(gt_x, gt_y, gt_w, gt_h, pred_x, pred_y, pred_w, pred_h, thr=0.7):
    if abs(gt_x - pred_x) < ((gt_w + pred_w) / 2.0) and abs(gt_y - pred_y) < ((gt_h + pred_h) / 2.0):
        lu_x = min((gt_x + (gt_w / 2.0)), (pred_x + (pred_w / 2.0)))
        lu_y = min((gt_y + (gt_h / 2.0)), (pred_y + (pred_h / 2.0)))

        rd_x = max((gt_x - (gt_w / 2.0)), (pred_x - (pred_w / 2.0)))
        rd_y = max((gt_y - (gt_h / 2.0)), (pred_y - (pred_h / 2.0)))

        w = abs(rd_x - lu_x)
        h = abs(rd_y - lu_y)

        inter_square = w * h
        union_square = (gt_h * gt_w) + (pred_h * pred_w) - inter_square
        IOU = inter_square / union_square
        if IOU > thr:
            return True
    return False


def match_detect(gt, pred, thr=0.7):
    # 输入的 gt 和 pred shape 为 X, 4
    # x, y, w, h
    
    # 这个地方还应该增加阈值,取除太小的区域
    gt = sorted(gt, key=lambda s: s[0])
    pred = sorted(pred, key=lambda s: s[0])
    count = 0
    for gt_label in gt:
        gt_x, gt_y, gt_w, gt_h = gt_label
        for idx, pred_label in enumerate(pred):
            pred_x, pred_y, pred_w, pred_h = pred_label
            if iou(gt_x, gt_y, gt_w, gt_h, pred_x, pred_y, pred_w, pred_h, thr=thr):
                count += 1
                np.delete(pred, idx, axis=0)
                break
    return count
            
