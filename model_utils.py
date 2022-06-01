''' Model Construction Utils '''
import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.mobilenet import MobileNet

from functools import wraps, reduce

L2_FACTOR = 1e-5


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')



@wraps(Conv2D)
def YoloConv2D(*args, **kwargs):
    """Wrapper to set Yolo parameters for Conv2D."""
    yolo_conv_kwargs = {'kernel_regularizer': l2(L2_FACTOR)}
    yolo_conv_kwargs['bias_regularizer'] = l2(L2_FACTOR)
    yolo_conv_kwargs.update(kwargs)
    #yolo_conv_kwargs = kwargs
    return Conv2D(*args, **yolo_conv_kwargs)


@wraps(DepthwiseConv2D)
def YoloDepthwiseConv2D(*args, **kwargs):
    """Wrapper to set Yolo parameters for DepthwiseConv2D."""
    yolo_conv_kwargs = {'kernel_regularizer': l2(L2_FACTOR)}
    yolo_conv_kwargs['bias_regularizer'] = l2(L2_FACTOR)
    yolo_conv_kwargs.update(kwargs)
    #yolo_conv_kwargs = kwargs
    return DepthwiseConv2D(*args, **yolo_conv_kwargs)


def CustomBatchNormalization(*args, **kwargs):
    if tf.__version__ >= '2.2':
        from tensorflow.keras.layers.experimental import SyncBatchNormalization
        BatchNorm = SyncBatchNormalization
    else:
        BatchNorm = BatchNormalization

    return BatchNorm(*args, **kwargs)



@wraps(YoloConv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for YoloConv2D."""
    darknet_conv_kwargs = {'padding': 'valid' if kwargs.get('strides')==(2,2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return YoloConv2D(*args, **darknet_conv_kwargs)



def Depthwise_Separable_Conv2D_BN_Leaky(filters, kernel_size=(3, 3), block_id_str=None):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    return compose(
        YoloDepthwiseConv2D(kernel_size, padding='same', name='conv_dw_' + block_id_str),
        CustomBatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_dw_%s_leaky_relu' % block_id_str),
        YoloConv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        CustomBatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_pw_%s_leaky_relu' % block_id_str))


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by CustomBatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        CustomBatchNormalization(),
        LeakyReLU(alpha=0.1))