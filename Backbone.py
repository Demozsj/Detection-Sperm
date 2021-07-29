from functools import wraps
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, Concatenate, Layer
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils import compose


@wraps(Conv2D)
def Conv2d(*args, **kwargs):
    conv_kwrags={'kernel_regularizer': l2(5e-4),
                 'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    conv_kwrags.update(kwargs)
    return Conv2D(*args, **conv_kwrags)


class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def Conv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs={'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(Conv2d(*args, **no_bias_kwargs), BatchNormalization(), Mish())


# Construction of residual structure
def resblock(inputs, num_filters, num_blocks, all_narrow=True):
    preconv1 = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    preconv1 = Conv2D_BN_Mish(num_filters, (3, 3), strides=(2, 2))(preconv1)
    shortconv = Conv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv1)
    mainconv = Conv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv1)
    for i in range(num_blocks):
        y = compose(
            Conv2D_BN_Mish(num_filters // 2, (1, 1)),
            Conv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (3, 3)))(mainconv)
        mainconv = Add()([mainconv, y])
    postconv = Conv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(mainconv)
    route = Concatenate()([postconv, shortconv])

    return Conv2D_BN_Mish(num_filters, (1, 1))(route)


# Build the backbone of TOD-CNN network
def network_body(inputs):
    inputs = Conv2D_BN_Mish(32, (3, 3))(inputs)
    inputs = resblock(inputs, 64, 1, False)
    inputs = resblock(inputs, 128, 2)
    inputs = resblock(inputs, 256, 8)
    feature_map1 = inputs
    inputs = resblock(inputs, 512, 8)
    feature_map2 = inputs
    inputs = resblock(inputs, 1024, 4)
    feature_map3 = inputs
    return feature_map1, feature_map2, feature_map3
