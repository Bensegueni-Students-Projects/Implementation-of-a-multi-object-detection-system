#!/usr/bin/python3
# -*- coding=utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

def yolo3_decode(feats, anchors, num_classes, input_shape, scale_x_y=None, calc_loss=False):
    """Decode final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    if scale_x_y:
        # Eliminate grid sensitivity trick involved in YOLOv4
        #
        # Reference Paper & code:
        #     "YOLOv4: Optimal Speed and Accuracy of Object Detection"
        #     https://arxiv.org/abs/2004.10934
        #     https://github.com/opencv/opencv/issues/17148
        #
        box_xy_tmp = K.sigmoid(feats[..., :2]) * scale_x_y - (scale_x_y - 1) / 2
        box_xy = (box_xy_tmp + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    else:
        box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo3_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    input_shape = K.cast(input_shape, K.dtype(box_xy))
    image_shape = K.cast(image_shape, K.dtype(box_xy))

    #reshape the image_shape tensor to align with boxes dimension
    image_shape = K.reshape(image_shape, [-1, 1, 1, 1, 2])

    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    # reverse offset/scale to match (w,h) order
    offset = offset[..., ::-1]
    scale = scale[..., ::-1]

    box_xy = (box_xy - offset) * scale
    box_wh *= scale

    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # x_min
        box_mins[..., 1:2],  # y_min
        box_maxes[..., 0:1],  # x_max
        box_maxes[..., 1:2]  # y_max
    ])

    # Scale boxes back to original image shape.
    image_wh = image_shape[..., ::-1]
    boxes *= K.concatenate([image_wh, image_wh])
    return boxes


def yolo3_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, scale_x_y):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo3_decode(feats,
        anchors, num_classes, input_shape, scale_x_y=scale_x_y)
    boxes = yolo3_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    # check if only 1 class for different score
    box_scores = tf.cond(K.equal(K.constant(value=num_classes, dtype='int32'), 1), lambda: box_confidence, lambda: box_confidence * box_class_probs)
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def get_anchorset(anchors, num_layers, l):
    if num_layers == 3: #YOLOv3 arch
        if l == 0:
            anchorset = anchors[6:]
        elif l == 1:
            anchorset = anchors[3:6]
        elif l == 2:
            anchorset = anchors[:3]
    elif num_layers == 2: # Tiny YOLOv3 arch
        if l == 0:
            anchorset = anchors[3:]
        elif l == 1:
            anchorset = anchors[:3]
    else:
        raise ValueError('Invalid layer number')

    return anchorset


def yolo3_postprocess(args,
              anchors,
              num_classes,
              max_boxes=100,
              confidence=0.1,
              iou_threshold=0.4,
              elim_grid_sense=False):
    """Postprocess for YOLOv3 model on given input and return filtered boxes."""

    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    image_shape = args[num_layers]

    # here we sort the prediction tensor list with grid size (e.g. 19/38/76)
    # to make sure it matches with anchors order
    yolo_outputs.sort(key=lambda x: x.shape[1])

    if num_layers == 3:
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]
    else:
        anchor_mask = [[3,4,5], [0,1,2]]
        scale_x_y = [1.05, 1.05] if elim_grid_sense else [None, None]

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32

    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo3_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape, scale_x_y=scale_x_y[l])
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def batched_yolo3_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, scale_x_y):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo3_decode(feats,
        anchors, num_classes, input_shape, scale_x_y=scale_x_y)

    num_anchors = len(anchors)
    grid_shape = K.shape(feats)[1:3] # height, width
    total_anchor_num = grid_shape[0] * grid_shape[1] * num_anchors

    boxes = yolo3_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, total_anchor_num, 4])
    # check if only 1 class for different score
    box_scores = tf.cond(K.equal(K.constant(value=num_classes, dtype='int32'), 1), lambda: box_confidence, lambda: box_confidence * box_class_probs)
    box_scores = K.reshape(box_scores, [-1, total_anchor_num, num_classes])
    return boxes, box_scores


def batched_yolo3_postprocess(args,
              anchors,
              num_classes,
              max_boxes=100,
              confidence=0.1,
              iou_threshold=0.4,
              elim_grid_sense=False):
    """Postprocess for YOLOv3 model on given input and return filtered boxes."""

    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    image_shape = args[num_layers]

    # here we sort the prediction tensor list with grid size (e.g. 19/38/76)
    # to make sure it matches with anchors order
    yolo_outputs.sort(key=lambda x: x.shape[1])

    if num_layers == 3:
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]
    else:
        anchor_mask = [[3,4,5], [0,1,2]]
        scale_x_y = [1.05, 1.05] if elim_grid_sense else [None, None]

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    batch_size = K.shape(image_shape)[0] # batch size, tensor

    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = batched_yolo3_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape, scale_x_y=scale_x_y[l])
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=1)
    box_scores = K.concatenate(box_scores, axis=1)

    mask = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    def single_image_nms(b, batch_boxes, batch_scores, batch_classes):
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(num_classes):
            # TODO: use keras backend instead of tf.
            class_boxes = tf.boolean_mask(boxes[b], mask[b, :, c])
            class_box_scores = tf.boolean_mask(box_scores[b, :, c], mask[b, :, c])
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)

        boxes_ = K.concatenate(boxes_, axis=0)
        scores_ = K.concatenate(scores_, axis=0)
        classes_ = K.concatenate(classes_, axis=0)

        batch_boxes = batch_boxes.write(b, boxes_)
        batch_scores = batch_scores.write(b, scores_)
        batch_classes = batch_classes.write(b, classes_)

        return b+1, batch_boxes, batch_scores, batch_classes

    batch_boxes = tf.TensorArray(K.dtype(boxes), size=1, dynamic_size=True)
    batch_scores = tf.TensorArray(K.dtype(box_scores), size=1, dynamic_size=True)
    batch_classes = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True)
    _, batch_boxes, batch_scores, batch_classes = tf.while_loop(lambda b,*args: b<batch_size, single_image_nms, [0, batch_boxes, batch_scores, batch_classes])

    batch_boxes = batch_boxes.stack()
    batch_scores = batch_scores.stack()
    batch_classes = batch_classes.stack()

    return batch_boxes, batch_scores, batch_classes