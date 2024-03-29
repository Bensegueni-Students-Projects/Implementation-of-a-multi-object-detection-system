''' YOLOv3 Bodies '''
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet

from yolo3_predictor import yolo3lite_predictor, tiny_yolo3_predictor, tiny_yolo3lite_predictor



def yolo3lite_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create YOLO_v3 Lite MobileNet model CNN body in keras.'''
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenet.layers)))

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x (1024*alpha)
    # conv_pw_11_relu :26 x 26 x (512*alpha)
    # conv_pw_5_relu : 52 x 52 x (256*alpha)

    # f1: 13 x 13 x (1024*alpha)
    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f2: 26 x 26 x (512*alpha)
    f2 = mobilenet.get_layer('conv_pw_11_relu').output
    # f3: 52 x 52 x (256*alpha)
    f3 = mobilenet.get_layer('conv_pw_5_relu').output

    f1_channel_num = int(1024*alpha)
    f2_channel_num = int(512*alpha)
    f3_channel_num = int(256*alpha)

    y1, y2, y3 = yolo3lite_predictor((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), num_anchors, num_classes)

    return Model(inputs = inputs, outputs=[y1,y2,y3])



def tiny_yolo3_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 MobileNet model CNN body in keras.'''
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenet.layers)))

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x (1024*alpha)
    # conv_pw_11_relu :26 x 26 x (512*alpha)
    # conv_pw_5_relu : 52 x 52 x (256*alpha)

    # f1: 13 x 13 x (1024*alpha)
    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f2: 26 x 26 x (512*alpha)
    f2 = mobilenet.get_layer('conv_pw_11_relu').output

    f1_channel_num = int(1024*alpha)
    f2_channel_num = int(512*alpha)

    y1, y2 = tiny_yolo3_predictor((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_mobilenet_body(inputs, num_anchors, num_classes, alpha=1.0):
    '''Create Tiny YOLO_v3 Lite MobileNet model CNN body in keras.'''
    mobilenet = MobileNet(input_tensor=inputs, weights='imagenet', include_top=False, alpha=alpha)
    print('backbone layers number: {}'.format(len(mobilenet.layers)))

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x (1024*alpha)
    # conv_pw_11_relu :26 x 26 x (512*alpha)
    # conv_pw_5_relu : 52 x 52 x (256*alpha)

    # f1: 13 x 13 x (1024*alpha)
    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f2: 26 x 26 x (512*alpha)
    f2 = mobilenet.get_layer('conv_pw_11_relu').output

    f1_channel_num = int(1024*alpha)
    f2_channel_num = int(512*alpha)

    y1, y2 = tiny_yolo3lite_predictor((f1, f2), (f1_channel_num, f2_channel_num), num_anchors, num_classes)

    return Model(inputs, [y1,y2])

