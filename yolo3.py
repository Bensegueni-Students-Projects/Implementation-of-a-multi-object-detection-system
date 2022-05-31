''' Real-Time Multi Object Detection System based on YOLOv3 '''
''' Realized by K. BENMOUSSA & F. FELLAH '''

from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Lambda
from PIL import Image

classes_path = os.path.join('configs', 'voc_classes.txt')
anchors_path = os.path.join('configs', 'tiny_yolo3_anchors.txt')
weights_path = os.path.join('weights', 'tiny_yolo3_mobilenet_lite_416_voc.h5')
class_thresh = 0.3
iou = 0.5
model_input_shape = (416, 416)



class YOLO():
    @classmethod

    def __init__(self):
        self.class_names = get_classes(classes_path)
        self.anchors = get_anchors(anchors_path)
        self.colors = get_colors(len(self.class_names))
        self.inference_model = self.generate_model()

    def generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(weights_path)
        assert weights_path.endswith('.h5'), 'Keras model weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #YOLOv3 model has 9 anchors and 3 feature layers but
        #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        #so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors//3

        inference_model = get_yolo3_inference_model(self.model_type, self.anchors, num_classes, weights_path=weights_path, input_shape=self.model_input_shape + (3,), confidence=self.score, iou_threshold=self.iou)

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
        print("Inference time: {:.8f} secs".format(end - start))

        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)

        out_classnames = [self.class_names[c] for c in out_classes]
        return Image.fromarray(image_array), out_boxes, out_classnames, out_scores

    def dump_model_file(self, output_model_file):
        self.inference_model.save(output_model_file)



def detect_video(yolo, video_path, output_path=""):
    import cv2
    
    vid = cv2.VideoCapture(0 if video_path == '0' else video_path)
    
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
        
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
        r_image, _, _, _ = yolo.detect_image(image)
        r_image.show()



def get_yolo3_inference_model(model_type, anchors, num_classes, weights_path=None, input_shape=None, confidence=0.1, iou_threshold=0.4):
    '''create the inference model, for YOLOv3'''
    num_anchors = len(anchors)
    #YOLOv3 model has 9 anchors and 3 feature layers but
    #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
    #so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors//3

    image_shape = Input(shape=(2,), dtype='int64', name='image_shape')

    model_body, _ = get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, input_shape=input_shape)
    print('Create {} YOLOv3 {} model with {} anchors and {} classes.'.format('Tiny' if num_feature_layers==2 else '', model_type, num_anchors, num_classes))

    if weights_path:
        model_body.load_weights(weights_path, by_name=False)
        print('Load weights {}.'.format(weights_path))

    boxes, scores, classes = Lambda(batched_yolo3_postprocess, name='yolo3_postprocess',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'confidence': confidence, 'iou_threshold': iou_threshold, 'elim_grid_sense': elim_grid_sense})(
        [*model_body.output, image_shape])
    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model


def get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, input_tensor=None, input_shape=None):
    #prepare input tensor
    if input_shape:
        input_tensor = Input(shape=input_shape, name='image_input')

    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

    #Tiny YOLOv3 model has 6 anchors and 2 feature layers
    if num_feature_layers == 2:
        if model_type in yolo3_tiny_model_map:
            model_function = yolo3_tiny_model_map[model_type][0]
            backbone_len = yolo3_tiny_model_map[model_type][1]
            weights_path = yolo3_tiny_model_map[model_type][2]

            if weights_path:
                model_body = model_function(input_tensor, num_anchors//2, num_classes, weights_path=weights_path)
            else:
                model_body = model_function(input_tensor, num_anchors//2, num_classes)
        else:
            raise ValueError('This model type is not supported now')

    #YOLOv3 model has 9 anchors and 3 feature layers
    elif num_feature_layers == 3:
        if model_type in yolo3_model_map:
            model_function = yolo3_model_map[model_type][0]
            backbone_len = yolo3_model_map[model_type][1]
            weights_path = yolo3_model_map[model_type][2]

            if weights_path:
                model_body = model_function(input_tensor, num_anchors//3, num_classes, weights_path=weights_path)
            else:
                model_body = model_function(input_tensor, num_anchors//3, num_classes)
        else:
            raise ValueError('This model type is not supported now')
    else:
        raise ValueError('model type mismatch anchors')

    return model_body, backbone_len



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



def draw_boxes(image, boxes, classes, scores, class_names, colors, show_score=True):
    if boxes is None or len(boxes) == 0:
        return image
    if classes is None or len(classes) == 0:
        return image

    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = map(int, box)

        class_name = class_names[cls]
        if show_score:
            label = '{} {:.2f}'.format(class_name, score)
        else:
            label = '{}'.format(class_name)
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
    # get wrapped inference object, you can also try "YOLO" here ;)
    yolo = YOLO()
    
    if 'image' in args:
        detect_img(yolo, args.image)
    elif 'video' in args:
        detect_video(yolo, args.video)
    else:
        print("Specify video input path or image input path")