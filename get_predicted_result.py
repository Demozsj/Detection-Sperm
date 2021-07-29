import os
import numpy as np
from keras import backend as K
from keras.models import load_model
from TOD_CNN import TODCNN_body, TODCNN_eval
from keras.layers import Input
import colorsys
from utils import letterbox_image
from PIL import Image, ImageFont, ImageDraw
import shutil


class TODCNN(object):
    _defaults = {
        "anchors_path": 'model_data/sperm_anchors.txt',
        "classes_path": 'model_data/sperm_classes.txt',
        "score": 0.5,
        "iou": 0.5,
        "model_image_size": (416, 416),
        "letterbox_image": False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, model_path, iou=0.5, confidence=0.5):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.model_path = model_path
        self.iou = iou
        self.score = confidence
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        self.model = TODCNN_body(Input(shape=(None, None, 3)), num_anchors, num_classes)
        self.model.load_weights(self.model_path)
        print('{} model, {} anchors, and {} classes loaded.'.format(model_path, num_anchors, num_classes))
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))

        boxes, scores, classes = TODCNN_eval(self.model.output, self.anchors,
                                             num_classes, self.input_image_shape,
                                             score_threshold=self.score, iou_threshold=0.25)
        return boxes, scores, classes

    def detect_image(self, image_id, image):
        f = open("./input/detection-results/" + image_id + ".txt", "w")
        boxed_image = letterbox_image(image, self.model_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[int(c)]
            score = str(out_scores[i])
            for p in range(4):
                if out_boxes[i][p] < 0:
                    out_boxes[i][p] = 0
            top, left, bottom, right = out_boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),
                                             str(int(bottom))))
        f.close()


    def detect_image_for_GUI(self, image):
        new_image_size = self.model_image_size
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image


def Get_DR(model_path):
    TOD_CNN = TODCNN(model_path)
    image_ids = open('ImageSet/Main/test.txt').read().strip().split()
    if os.path.exists("./input/detection-results"):
        shutil.rmtree("./input/detection-results")
    if os.path.exists("./input/images-optional"):
        shutil.rmtree("./input/images-optional")
    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/detection-results"):
        os.makedirs("./input/detection-results")
    if not os.path.exists("./input/images-optional"):
        os.makedirs("./input/images-optional")

    for image_id in image_ids:
        image_path = "./ImageSet/PNGImages/Img/" + image_id + ".png"
        image = Image.open(image_path)
        image.save("./input/images-optional/" + image_id + ".png")
        TOD_CNN.detect_image(image_id, image)
        print(image_id, " done!")
    print("Conversion completed!")
