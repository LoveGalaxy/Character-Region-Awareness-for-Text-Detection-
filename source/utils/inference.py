import torch
import cv2
import numpy as np

def get_detct_box(gre, aff, thread_g=0.1, thread_a=0.1):
    # simple imple
    m = (gre > thread_g) + (aff > thread_a).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m, 4, cv2.CV_32S)
    return num_labels, labels, stats, centroids

if __name__ == "__main__":
    pass