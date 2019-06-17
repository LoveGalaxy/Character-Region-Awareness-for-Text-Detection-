import torch
import numpy as np
import cv2

def box_center(points):
    """
        support two input ways
        4 points: x1, y1, x2, y2, x3, y3, x4, y4
        2 points: lt_x1, lt_y1, rd_x2, rd_y2
    """
    if len(points) == 4:
        x1, y1, x2, y2 = points
        x3, y3, x4, y4 = x2, y1, x1, y2
    elif len(points) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = points
    else:
        raise("please input 2 points or 4 points, check it")
    center_x = round((x1 + x2 + x3 + x4) / 4)
    center_y = round((y1 + y2 + y3 + y4) / 4)
    return center_x, center_y, x1, y1, x3, y3, x2, y2, x4, y4

def triangle_center(points):
    if len(points) == 6:
        x1, y1, x2, y2, x3, y3 = points
    else:
        raise("please input 3 points, check it") 
    center_x = round((x1 + x2 + x3) / 3)
    center_y = round((y1 + y2 + y3) / 3)
    return center_x, center_y

def sorted_boxes(boxes):
    # sorted by the left top point's x location
    boxes = sorted(boxes, key=lambda box:box[0])
    return boxes

def create_affine_boxes(boxes):
    affine_boxes = []
    if len(boxes) == 1:
        return affine_boxes
    for boxes_1, boxes_2 in zip(boxes[:-1], boxes[1:]):
        center_x1, center_y1, x1, y1, x3, y3, x2, y2, x4, y4 = box_center(boxes_1)
        points_x1, points_y1 = triangle_center([center_x1, center_y1, x1, y1, x2, y2])
        points_x2, points_y2 = triangle_center([center_x1, center_y1, x3, y3, x4, y4])
        center_x2, center_y2, x1, y1, x3, y3, x2, y2, x4, y4 = box_center(boxes_2)
        points_x3, points_y3 = triangle_center([center_x2, center_y2, x1, y1, x2, y2])
        points_x4, points_y4 = triangle_center([center_x2, center_y2, x3, y3, x4, y4])
        affine_boxes.append([points_x1, points_y1, points_x3, points_y3, points_x4, points_y4, points_x2, points_y2,])
    return affine_boxes

def find_min_rectangle(points):
    if len(points) == 4:
        x1, y1, x2, y2 = points
        x3, y3, x4, y4 = x2, y1, x1, y2
    elif len(points) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = points
    else:
        raise("please input 2 points or 4 points, check it")
    lt_x = min(x1, x2, x3, x4)
    lt_y = min(y1, y2, y3, y4)
    rd_x = max(x1, x2, x3, x4)
    rd_y = max(y1, y2, y3, y4)
    return np.float32([[lt_x, lt_y], [rd_x, lt_y], [rd_x, rd_y], [lt_x, rd_y]]), int(rd_x - lt_x), int(rd_y - lt_y)

def gaussian_kernel_2d_opencv(kernel_size = (3, 3)):

    ky = cv2.getGaussianKernel(kernel_size[0], int(kernel_size[0] / 4))
    kx = cv2.getGaussianKernel(kernel_size[1], int(kernel_size[1] / 4))
    return np.multiply(ky, np.transpose(kx))  

def aff_gaussian(gaussian, box, pts, deta_x, deta_y):
    de_x, de_y = box[0]
    box = box - [de_x, de_y]
    pts = pts - [de_x, de_y]
    M = cv2.getPerspectiveTransform(box, pts)
    res = cv2.warpPerspective(gaussian, M, (deta_y, deta_x))
    return res


def rotate(angle, image):
    
    h, w = image.shape[1:]
    image = image.transpose((1, 2, 0))

    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    image = image.transpose((2, 0, 1))

    return image, M

def rotate_point(M, x, y):
    point = np.array([x, y, 1])
    x, y = M.dot(point)
    return x, y


if __name__ == "__main__":
    boxes = [[0, 0, 5, 5], [5, 0, 10, 15]] #, [2, 1, 4, 3], [1, 3, 4, 5], [5, 2, 5, 4], [3, 4, 4, 6]]
    boxes = sorted_boxes(boxes)
    print(boxes)
    a_boxes = create_affine_boxes(boxes)
    print(a_boxes)
    box, deta_x, deta_y = find_min_rectangle(a_boxes[0])
    gaussian = gaussian_kernel_2d_opencv(kernel_size=(deta_x, deta_y))
    print(gaussian)
    pts = np.float32([[a_boxes[0][0], a_boxes[0][1]], [a_boxes[0][2], a_boxes[0][3]], [a_boxes[0][6], a_boxes[0][7]], [a_boxes[0][4], a_boxes[0][5]]])
    M = cv2.getPerspectiveTransform(box, pts)
    res = cv2.warpPerspective(gaussian, M, (deta_y, deta_x))
    print(res)
    