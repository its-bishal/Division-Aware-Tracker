
import cv2
import math
import numpy as np
from skimage.measure import label, regionprops  # type: ignore
from skimage.morphology import square, erosion, dilation  # type: ignore
from skimage import segmentation  # type: ignore


def region_to_mask(region, height, width):
    coords = region.coords
    mask = np.zeros([height, width])
    mask[coords[:, 0], coords[:, 1]] = 1
    mask = dilation(mask, square(1))  # 3 -> 1
    mask = np.tile(np.expand_dims(mask, 2), (1, 1, 3))
    return np.clip(mask, a_min=0, a_max=1).astype(np.float32)


def get_region(mask, dilation_width=None, erosion_width=None):
    if dilation_width:
        mask = dilation(mask, square(dilation_width))  # 3 -> 7
    if erosion_width:
        mask = erosion(mask, square(erosion_width))  # 3 -> 7
    individual_mask = label(mask, connectivity=2)  # 2 -> 1
    return regionprops(individual_mask)


def get_region_with_rw(mask, old_center_points=None):
    mask = erosion(mask, square(3))  # 3 -> 7
    center_points = center_point(mask, use_dilation=False)
    if old_center_points is not None:
        center_points += old_center_points
    markers = label(center_points)
    markers[mask < 0.5] = -1
    individual_mask = segmentation.random_walker(mask, markers)
    return regionprops(individual_mask), old_center_points


def center_point(mask, use_dilation=True):
    v, h = mask.shape
    center_mask = np.zeros([v, h])
    mask = erosion(mask, square(3))
    individual_mask = label(mask, connectivity=2)
    prop = regionprops(individual_mask)
    for cordinates in prop:
        temp_center = cordinates.centroid
        if not math.isnan(temp_center[0]) and not math.isnan(temp_center[1]):
            temp_mask = np.zeros([v, h])
            temp_mask[int(temp_center[0]), int(temp_center[1])] = 1
            if not use_dilation:
                center_mask += temp_mask
            else:
                center_mask += dilation(temp_mask, square(2))
    return np.clip(center_mask, a_min=0, a_max=1).astype(np.uint8)


def bbox_mark(mask):
    v, h = mask.shape
    bbox_mask = np.zeros([v, h])
    mask = erosion(mask, square(3))
    individual_mask = label(mask, connectivity=2)
    prop = regionprops(individual_mask)
    for cordinates in prop:
        temp_bbox = list(cordinates.bbox)
        temp_bbox[0] = np.maximum(temp_bbox[0] - 1, 0)
        temp_bbox[1] = np.maximum(temp_bbox[1] - 1, 0)
        temp_bbox[2] = np.minimum(temp_bbox[2] - 1, v)
        temp_bbox[3] = np.minimum(temp_bbox[3] - 1, h)
        if temp_bbox[0] == h or temp_bbox[1] == v:
            continue
        if not math.isnan(temp_bbox[0]) and not math.isnan(temp_bbox[2]):
            temp_mask = np.zeros([v, h])
            temp_mask[int(temp_bbox[0]):int(
                temp_bbox[2]), int(temp_bbox[1])] = 1
            temp_mask[int(temp_bbox[0]):int(
                temp_bbox[2]), int(temp_bbox[3])] = 1
            temp_mask[int(temp_bbox[0]), int(
                temp_bbox[1]):int(temp_bbox[3])] = 1
            temp_mask[int(temp_bbox[2]), int(
                temp_bbox[1]):int(temp_bbox[3])] = 1
            bbox_mask += dilation(temp_mask, square(1))
    return np.clip(bbox_mask, a_min=0, a_max=1).astype(np.uint8)


def draw_individual_edge(mask):
    v, h = mask.shape
    edge = np.zeros([v, h])
    individual_mask = label(mask, connectivity=2)
    for index in np.unique(individual_mask):
        if index == 0:
            continue
        temp_mask = np.copy(individual_mask)
        temp_mask[temp_mask != index] = 0
        temp_mask[temp_mask == index] = 1
        temp_mask = dilation(temp_mask, square(3))
        temp_edge = cv2.Canny(temp_mask.astype(np.uint8), 2, 5)/255
        edge += temp_edge
    return np.clip(edge, a_min=0, a_max=1).astype(np.uint8)


def center_edge(mask, image):
    center_map = center_point(mask)
    bbox_map = bbox_mark(mask)
    edge_map = draw_individual_edge(mask)
    comb_mask = center_map + edge_map + bbox_map
    comb_mask = np.clip(comb_mask, a_min=0, a_max=1)
    check_image = np.copy(image)
    comb_mask *= 255
    check_image[:, :, 0] = np.maximum(check_image[:, :, 0], comb_mask)
    return check_image.astype(np.uint8), comb_mask.astype(np.uint8)



def mask_refine(image, mask):
    # img = np.tile(np.expand_dims(np.copy(mask),2),(1,1,3))
    color_mask = np.tile(mask, (1, 1, 3))
    kernel = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imwrite('Opening_Image.jpg', opening)

    # sure_bg = cv2.erode(opening, kernel,iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=5)
    # cv2.imshow('Background Image', sure_bg)
    cv2.imwrite('Background_Image.jpg', sure_bg)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.5*dist_transform.max(), 255, 0)
    cv2.imwrite('Foreground_Image.jpg', sure_fg)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imwrite('Unknown_Image.jpg', unknown)
    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers+1
    markers[unknown == 255] = 0
    markers = cv2.watershed(color_mask, markers)
    labels = np.unique(markers)

    check_image = np.tile(np.expand_dims(np.copy(image), axis=2), (1, 1, 3))
    area = [(markers == x).sum() for x in labels]
    bg = labels[area.index(max(area))]
    for lbl in labels:
        if lbl == -1:
            color_mask[markers == lbl] = [0, 0, 255]
        if lbl == bg:
            color_mask[markers == lbl] = [0, 0, 0]
        else:
            color = tuple((50 + np.random.rand(3) * 255).astype(int).tolist())
            color_mask[markers == lbl] = color
            add_color = np.maximum(check_image, np.ones_like(
                check_image) * np.array(color))[markers == lbl]
            check_image[markers == lbl] = add_color
    # cv2.imshow('Result Image', img)
    cv2.imwrite('Result_Image.jpg', color_mask)
    return color_mask, check_image.astype(np.uint8)