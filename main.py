import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
from make_txt import first_step, second_step
from TOD_CNN import TODCNN_body
from loss_function import loss_func
from utils import get_random_data, get_random_data_with_Mosaic, WarmUpCosineDecayScheduler
from get_GroundTruth import Get_GT
from get_predicted_result import Get_DR
from calculate_mAP import Get_map


# Set the parameters of the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, mosaic=False):
    n = len(annotation_lines)
    i = 0
    flag = True
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            if mosaic:
                if flag and (i + 4) < n:
                    image, box = get_random_data_with_Mosaic(annotation_lines[i:i + 4], input_shape)
                    i = (i + 1) % n
                else:
                    image, box = get_random_data(annotation_lines[i], input_shape)
                    i = (i + 1) % n
                flag = bool(1 - flag)
            else:
                image, box = get_random_data(annotation_lines[i], input_shape)
                i = (i + 1) % n
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = 1
    anchor_mask = [[0, 1, 2, 3, 4, 5]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')  # 416,416

    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    true_boxes[..., 0:2] = boxes_xy / input_shape[:]
    true_boxes[..., 2:4] = boxes_wh / input_shape[:]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


if __name__ == '__main__':
    first_step()  # Write the path of training set, verification set and test set to TXT file
    second_step()  # Read the data information in the files of training set, verification set and test set
    # Set the appropriate file path
    train_annotation_path = 'train.txt'
    val_annotation_path = 'val.txt'
    classes_path = 'model_data/sperm_classes.txt'
    anchors_path = 'model_data/sperm_anchors.txt'
    weights_path = 'model_data/sperm_weight.h5'
    # Get the category and anchors
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)
    log_dir = 'logs/'
    input_shape = (416, 416)
    # Sets whether to use Mosaic, Cosine_scheduler and Cosine_scheduler
    mosaic = True
    Cosine_scheduler = True
    label_smoothing = 0.05

    K.clear_session()
    # Set the input, output, network structure loss function and weight loading of the model (transfer learning).
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    model_body = TODCNN_body(image_input, num_anchors, num_classes)
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    y_true = [Input(shape=(h // {0: 8}[l], w // {0: 8}[l], num_anchors, num_classes + 5)) for l in range(1)]
    loss_input = [model_body.output, *y_true]
    model_loss = Lambda(loss_func, output_shape=(1,), name='TODCNN_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,
                                   'label_smoothing': label_smoothing})(loss_input)

    model = Model([model_body.input, *y_true], model_loss)

    logging = TensorBoard(log_dir=log_dir)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    # Disorder the training set and the verification set
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.shuffle(val_lines)
    num_val = int(len(val_lines))
    num_train = int(len(train_lines))
    # Freeze the backbone of the model and fine-tune the rest of the model
    freeze_layers = 250
    for i in range(freeze_layers):
        model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
    # Start training
    if True:
        Init_epoch = 0
        Freeze_epoch = 50
        batch_size = 16
        learning_rate_base = 1e-3
        if Cosine_scheduler:
            warmup_epoch = int((Freeze_epoch - Init_epoch) * 0.2)
            total_steps = int((Freeze_epoch - Init_epoch) * num_train / batch_size)
            warmup_steps = int(warmup_epoch * num_train / batch_size)
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                   total_steps=total_steps,
                                                   warmup_learning_rate=1e-4,
                                                   warmup_steps=warmup_steps,
                                                   hold_base_rate_steps=num_train,
                                                   min_learn_rate=1e-6
                                                   )
            model.compile(optimizer=Adam(), loss={'TODCNN_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
            model.compile(optimizer=Adam(learning_rate_base), loss={'TODCNN_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator(train_lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(val_lines[:num_val], batch_size, input_shape, anchors, num_classes,
                                           mosaic=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=Freeze_epoch,
            initial_epoch=Init_epoch,
            callbacks=[logging, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    for i in range(freeze_layers):
        model_body.layers[i].trainable = True

    if True:
        Freeze_epoch = 50
        Epoch = 150
        batch_size = 4
        learning_rate_base = 1e-4
        if Cosine_scheduler:
            warmup_epoch = int((Epoch - Freeze_epoch) * 0.2)
            total_steps = int((Epoch - Freeze_epoch) * num_train / batch_size)
            warmup_steps = int(warmup_epoch * num_train / batch_size)
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                   total_steps=total_steps,
                                                   warmup_learning_rate=1e-5,
                                                   warmup_steps=warmup_steps,
                                                   hold_base_rate_steps=num_train // 2,
                                                   min_learn_rate=1e-6
                                                   )
            model.compile(optimizer=Adam(), loss={'TODCNN_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
            model.compile(optimizer=Adam(learning_rate_base), loss={'TODCNN_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator(train_lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(val_lines[:num_val], batch_size, input_shape, anchors, num_classes,
                                           mosaic=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=Epoch,
            initial_epoch=Freeze_epoch,
            callbacks=[logging, reduce_lr, early_stopping, checkpoint])
        model.save_weights('./model_data/sperm_last.h5')
    Get_GT()  # get ground truth results
    Get_DR('./model_data/sperm_last.h5')  # get predicted results
    Get_map()  # Calculate the relevant metrics of the model
