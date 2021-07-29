from functools import wraps
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from Backbone import network_body
from utils import compose


@wraps(Conv2D)
def Conv2d(*args, **kwargs):
    conv_kwrags={'kernel_regularizer': l2(5e-4),
                 'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    conv_kwrags.update(kwargs)
    return Conv2D(*args, **conv_kwrags)


def Conv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(Conv2d(*args, **no_bias_kwargs), BatchNormalization(), LeakyReLU(alpha=0.1))


def make_five_convs(inputs, num_filters):
    # Five times convolution
    inputs = Conv2D_BN_Leaky(num_filters, (1,1))(inputs)
    inputs = Conv2D_BN_Leaky(num_filters*2, (3,3))(inputs)
    inputs = Conv2D_BN_Leaky(num_filters, (1,1))(inputs)
    inputs = Conv2D_BN_Leaky(num_filters*2, (3,3))(inputs)
    inputs = Conv2D_BN_Leaky(num_filters, (1,1))(inputs)
    return inputs


# Build the overall structure of TOD-CNN
def TODCNN_body(inputs, num_anchors, num_classes):
    f1, f2, f3 = network_body(inputs)
    F = Conv2D_BN_Leaky(512, (1, 1))(f3)
    F = Conv2D_BN_Leaky(1024, (3, 3))(F)
    F = Conv2D_BN_Leaky(512, (1, 1))(F)
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(F)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(F)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(F)
    F = Concatenate()([maxpool1, maxpool2, maxpool3, F])
    F = Conv2D_BN_Leaky(512, (1, 1))(F)
    F = Conv2D_BN_Leaky(1024, (3, 3))(F)
    F = Conv2D_BN_Leaky(512, (1, 1))(F)
    F_upsample = compose(Conv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(F)
    F = Conv2D_BN_Leaky(256, (1, 1))(f2)
    F = Concatenate()([F, F_upsample])
    F = make_five_convs(F, 256)
    F_upsample = compose(Conv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(F)
    F = Conv2D_BN_Leaky(128, (1, 1))(f1)
    F = Concatenate()([F, F_upsample])
    F = make_five_convs(F, 128)
    output = Conv2D_BN_Leaky(256, (3, 3))(F)
    output = Conv2D(num_anchors * (num_classes + 5), (1, 1))(output)
    return Model(inputs, [output])


def model_head(outputs, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    grid_shape = K.shape(outputs)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(outputs))
    outputs = K.reshape(outputs, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    box_xy = (K.sigmoid(outputs[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(outputs))
    box_wh = K.exp(outputs[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(outputs))
    box_confidence = K.sigmoid(outputs[..., 4:5])
    box_class_probs = K.sigmoid(outputs[..., 5:])
    if calc_loss:
        return grid, outputs, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def get_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxes[..., 0:1],
        box_maxes[..., 1:2]
    ])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def get_boxes_and_scores(outputs, anchors, num_classes, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = model_head(outputs, anchors, num_classes, input_shape)
    boxes = get_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def TODCNN_eval(model_output, anchors, num_classes, image_shape, max_boxes=80, score_threshold=0.5, iou_threshold=0.3):
    anchor_mask = [[0, 1, 2, 3, 4, 5]]
    input_shape = K.shape(model_output)[1:3] * 8
    boxes = []
    box_scores = []
    _boxes, _box_scores = get_boxes_and_scores(model_output, anchors[anchor_mask[0]], num_classes, input_shape,
                                                image_shape)
    boxes.append(_boxes)
    box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
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
