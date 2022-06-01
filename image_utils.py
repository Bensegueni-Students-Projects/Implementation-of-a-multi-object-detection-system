"""Image processing functions."""
import numpy as np
import random
import math
import cv2
from enum import Enum
from PIL import Image



labelType = Enum('labelType', ('LABEL_TOP_OUTSIDE',
                               'LABEL_BOTTOM_OUTSIDE',
                               'LABEL_TOP_INSIDE',
                               'LABEL_BOTTOM_INSIDE',))



def letterbox_resize(image, target_size, return_padding_info=False):
    """
    Resize image with unchanged aspect ratio using padding

    # Arguments
        image: origin image to be resize
            PIL Image object containing image data
        target_size: target image size,
            tuple of format (width, height).
        return_padding_info: whether to return padding size & offset info
            Boolean flag to control return value

    # Returns
        new_image: resized PIL Image object.

        padding_size: padding image size (keep aspect ratio).
            will be used to reshape the ground truth bounding box
        offset: top-left offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    src_w, src_h = image.size
    target_w, target_h = target_size

    # calculate padding scale and padding offset
    scale = min(target_w/src_w, target_h/src_h)
    padding_w = int(src_w * scale)
    padding_h = int(src_h * scale)
    padding_size = (padding_w, padding_h)

    dx = (target_w - padding_w)//2
    dy = (target_h - padding_h)//2
    offset = (dx, dy)

    # create letterbox resized image
    image = image.resize(padding_size, Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128,128,128))
    new_image.paste(image, offset)

    if return_padding_info:
        return new_image, padding_size, offset
    else:
        return new_image

    

def reshape_boxes(boxes, src_size, target_size, padding_size, offset, horizontal_flip=False, vertical_flip=False):
    """
    Reshape bounding boxes from src_size image to target_size image,
    usually for training data preprocess

    # Arguments
        boxes: Ground truth object bounding boxes,
            numpy array of shape (num_boxes, 5),
            box format (xmin, ymin, xmax, ymax, cls_id).
        src_size: origin image size,
            tuple of format (width, height).
        target_size: target image size,
            tuple of format (width, height).
        padding_size: padding image shape,
            tuple of format (width, height).
        offset: top-left offset when padding target image.
            tuple of format (dx, dy).
        horizontal_flip: whether to do horizontal flip.
            boolean flag.
        vertical_flip: whether to do vertical flip.
            boolean flag.

    # Returns
        boxes: reshaped bounding box numpy array
    """
    if len(boxes)>0:
        src_w, src_h = src_size
        target_w, target_h = target_size
        padding_w, padding_h = padding_size
        dx, dy = offset

        # shuffle and reshape boxes
        np.random.shuffle(boxes)
        boxes[:, [0,2]] = boxes[:, [0,2]]*padding_w/src_w + dx
        boxes[:, [1,3]] = boxes[:, [1,3]]*padding_h/src_h + dy
        # horizontal flip boxes if needed
        if horizontal_flip:
            boxes[:, [0,2]] = target_w - boxes[:, [2,0]]
        # vertical flip boxes if needed
        if vertical_flip:
            boxes[:, [1,3]] = target_h - boxes[:, [3,1]]

        # check box coordinate range
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > target_w] = target_w
        boxes[:, 3][boxes[:, 3] > target_h] = target_h

        # check box width and height to discard invalid box
        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(boxes_w>1, boxes_h>1)] # discard invalid box

    return boxes




def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates



def normalize_image(image):
    image = image.astype(np.float32) / 255.0
    return image


def denormalize_image(image):
    image = (image * 255.0).astype(np.uint8)
    return image


def preprocess_image(image, model_input_shape):
    #resized_image = cv2.resize(image, model_input_shape[::-1], cv2.INTER_AREA)
    resized_image = letterbox_resize(image, model_input_shape[::-1])
    image_data = np.asarray(resized_image).astype('float32')
    image_data = normalize_image(image_data)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data



def draw_label(image, text, color, coords, label_type=labelType.LABEL_TOP_OUTSIDE):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    if label_type == labelType.LABEL_TOP_OUTSIDE or label_type == labelType.LABEL_BOTTOM_INSIDE:
        cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
        cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                    fontScale=font_scale,
                    color=(255, 255, 255),
                    lineType=cv2.LINE_AA)
    else: # LABEL_BOTTOM_OUTSIDE or LABEL_TOP_INSIDE
        cv2.rectangle(image, (x, y), (x + rect_width, y + rect_height), color, cv2.FILLED)
        cv2.putText(image, text, (x + padding, y + text_height + padding), font,
                    fontScale=font_scale,
                    color=(255, 255, 255),
                    lineType=cv2.LINE_AA)

    return image



def draw_boxes(image, boxes, classes, scores, class_names, colors):
    if boxes is None or len(boxes) == 0:
        return image
    if classes is None or len(classes) == 0:
        return image

    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = map(int, box)

        class_name = class_names[cls]
        label = '{} : {:.2f}%'.format(class_name, score*100)
        #print(label, (xmin, ymin), (xmax, ymax))

        # if no color info, use black(0,0,0)
        if colors == None:
            color = (0,0,0)
        else:
            color = colors[cls]

        # choose label type according to box size
        if ymin > 20:
            label_coords = (xmin, ymin)
            label_type = label_type=labelType.LABEL_TOP_OUTSIDE
        elif ymin <= 20 and ymax <= image.shape[0] - 20:
            label_coords = (xmin, ymax)
            label_type = label_type=labelType.LABEL_BOTTOM_OUTSIDE
        elif ymax > image.shape[0] - 20:
            label_coords = (xmin, ymin)
            label_type = label_type=labelType.LABEL_TOP_INSIDE

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)
        image = draw_label(image, label, color, label_coords, label_type)

    return image