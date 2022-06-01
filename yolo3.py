''' Real-Time Multi Object Detection System based on YOLOv3 '''
''' Realized by K. BENMOUSSA & F. FELLAH '''

import time
import cv2, colorsys
import os, argparse
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Lambda
from yolo3_body import yolo3lite_mobilenet_body, tiny_yolo3_mobilenet_body, tiny_yolo3lite_mobilenet_body
from postprocess import batched_yolo3_postprocess
from image_utils import draw_boxes, preprocess_image
from PIL import Image

classes_path = os.path.join('configs', 'voc_classes.txt')
anchors_path = os.path.join('configs', 'tiny_yolo3_anchors.txt')
weights_path = os.path.join('models', 'tiny_yolo3_mobilenet_lite_416_voc.h5')
iou_thresh = 0.5
class_thresh = 0.3
model_input_shape = (416, 416)



class YOLO():
    @classmethod

    def __init__(self):
        self.class_names = get_classes(classes_path)
        self.anchors = get_anchors(anchors_path)
        self.colors = get_colors(len(self.class_names))
        self.iou_thresh = iou_thresh
        self.class_thresh = class_thresh
        self.model_input_shape = model_input_shape
        self.inference_model = self.generate_model(self)

    def generate_model(self):
        '''to generate the bounding boxes'''
        self.weights_path = os.path.expanduser(weights_path)
        assert self.weights_path.endswith('.h5'), 'Keras model weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        num_feature_layers = num_anchors//3

        inference_model = get_yolo3_inference_model(self.anchors, num_classes, weights_path=self.weights_path, input_shape=self.model_input_shape + (3,), confidence=self.class_thresh, iou_thresh=self.iou_thresh)

        inference_model.summary()
        return inference_model

    def predict(self, image_data, image_shape):
        out_boxes, out_scores, out_classes = self.inference_model.predict([image_data, image_shape])

        out_boxes = out_boxes[0]
        out_scores = out_scores[0]
        out_classes = out_classes[0]

        out_boxes = out_boxes.astype(np.int32)
        out_classes = out_classes.astype(np.int32)
        return out_boxes, out_classes, out_scores

    def detect_image(self, image):
        if self.model_input_shape != (None, None):
            assert self.model_input_shape[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_input_shape[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_input_shape)

        # prepare origin image shape, (height, width) format
        image_shape = np.array([image.size[1], image.size[0]])
        image_shape = np.expand_dims(image_shape, 0)

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        end = time.time()
        print('Found {} boxes in {}'.format(len(out_boxes), 'the image'))
        print("Inference time: {:.4f} secs".format(end - start))

        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)

        out_classnames = [self.class_names[c] for c in out_classes]
        return Image.fromarray(image_array), out_boxes, out_classnames, out_scores

#    def dump_model_file(self, output_model_file):
#        self.inference_model.save(output_model_file)



def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def get_colors(number, bright=True):
    """
    Generate random colors for drawing bounding boxes.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    if number <= 0:
        return []

    brightness = 1.0 if bright else 0.7
    hsv_tuples = [(x / number, 1., brightness)
                  for x in range(number)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors




def get_yolo3_inference_model(anchors, num_classes, weights_path=None, input_shape=None, confidence=0.3, iou_thresh=0.5):
    num_anchors = len(anchors)
    num_feature_layers = num_anchors//3
    image_shape = Input(shape=(2,), dtype='int64', name='image_shape')
    input_tensor = Input(shape=input_shape, name='image_input')
    model_body = tiny_yolo3lite_mobilenet_body(input_tensor, num_anchors//2, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if weights_path:
        model_body.load_weights(weights_path, by_name=False)
        print('Load weights {}.'.format(weights_path))

    boxes, scores, classes = Lambda(batched_yolo3_postprocess, name='yolo3_postprocess',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'confidence': confidence, 'iou_threshold': iou_thresh})(
        [*model_body.output, image_shape])
    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model




def detect_video(yolo, video_path):
    vid = cv2.VideoCapture(0 if video_path == '0' else video_path)
    
    if not vid.isOpened():
        raise IOError("Can't access webcam or video !")
        
    print("CAP_PROP_FPS : '{}'".format(vid.get(cv2.CAP_PROP_FPS)))
    print("CAP_PROP_POS_MSEC : '{}'".format(vid.get(cv2.CAP_PROP_POS_MSEC)))
    print("CAP_PROP_FRAME_COUNT  : '{}'".format(vid.get(cv2.CAP_PROP_FRAME_COUNT)))

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    count = 0
    while True:
        ret, frame = vid.read()

        if ret != True:
            print('Stream end !')
            break

        image = Image.fromarray(frame)
        image, _, _, _ = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release everything if job is finished
    vid.release()
    cv2.destroyAllWindows()


def detect_img(yolo, image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except:
        print('Cannot open image file !')
    else:
        res, _, _, _ = yolo.detect_image(image)
        res.show()
        
        
        

def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, 
                                     description='YOLO - Multi-object detection system - BENMOUUSSA & FELLAH')
    
    parser.add_argument(
        "--image", nargs='?', type=str, required=False, default='',
        help = "Image\Video input path"
    )
    
    parser.add_argument(
        "--video", nargs='?', type=str, required=False, default='0',
        help = "Image\Video input path"
    )
    
    args = parser.parse_args()
    # get inference object
    yolo = YOLO()
    
    if args.image :
        detect_img(yolo, args.image)
    elif args.video :
        detect_video(yolo, args.video)
    else:
        print("Specify video input path or image input path")
        
        
if __name__ == '__main__':
    main()
