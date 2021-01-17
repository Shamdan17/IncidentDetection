#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers

layers = tf.keras.layers

def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return regularizers.l2(l2_weight_decay) if use_l2_regularizer else None


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   use_l2_regularizer=True,
                   batch_norm_decay=0.9,
                   batch_norm_epsilon=1e-5):
    """The identity block is the block that has no conv layer at shortcut.
    Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.
    batch_norm_decay: Moment of batch norm layers.
    batch_norm_epsilon: Epsilon of batch borm layers.
    Returns:
    Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 1
    
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
    conv_name_base = "layer" + str(stage) + '.'  + block + '.conv'
    bn_name_base = "layer" + str(stage) + '.'  + block + '.bn'

    x = layers.Conv2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '1')(
          input_tensor)
    x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '1')(
          x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
      filters2,
      kernel_size,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2')(
          x)
    x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2')(
          x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '3')(
          x)
    x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '3')(
          x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               use_l2_regularizer=True,
               batch_norm_decay=0.9,
               batch_norm_epsilon=1e-5):
    """A block that has a conv layer at shortcut.
    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.
    batch_norm_decay: Moment of batch norm layers.
    batch_norm_epsilon: Epsilon of batch borm layers.
    Returns:
    Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    
    bn_axis = 1
    
    conv_name_base = "layer" + str(stage) + '.'  + block + '.conv'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_name_base = "layer" + str(stage) + '.'  + block + '.bn'
    shrt_conv_name_base = "layer" + str(stage) + '.'  + block + '.downsample.0'
    shrt_bn_name_base = "layer" + str(stage) + '.'  + block + '.downsample.1'

    x = layers.Conv2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '1')(
          input_tensor)
    x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '1')(
          x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
      filters2,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2')(
          x)
    x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2')(
          x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '3')(
          x)
    x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '3')(
          x)

    shortcut = layers.Conv2D(
      filters3, (1, 1),
      strides=strides,
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=shrt_conv_name_base)(
          input_tensor)

    shortcut = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=shrt_bn_name_base)(
          shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def trunk(   batch_size=None,
             use_l2_regularizer=True,
             batch_norm_decay=0.9,
             batch_norm_epsilon=1e-10):
    """Instantiates the ResNet50 architecture.
    Args:
    batch_size: Size of the batches for each step.
    use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
    rescale_inputs: whether to rescale inputs from 0 to 1.
    batch_norm_decay: Moment of batch norm layers.
    batch_norm_epsilon: Epsilon of batch borm layers.
    Returns:
      A Keras model instance.
    """
    input_shape = (3, 224, 224)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    x = img_input

    bn_axis = 1
    
    block_config = dict(
      use_l2_regularizer=use_l2_regularizer,
      batch_norm_decay=batch_norm_decay,
      batch_norm_epsilon=batch_norm_epsilon)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(
      64, (7, 7),
      strides=(2, 2),
      padding='valid',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='conv1')(
          x)
    x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn1')(
          x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = conv_block(
      x, 3, [64, 64, 256], stage=1, block='0', strides=(1, 1), **block_config)
    x = identity_block(x, 3, [64, 64, 256], stage=1, block='1', **block_config)
    x = identity_block(x, 3, [64, 64, 256], stage=1, block='2', **block_config)

    x = conv_block(x, 3, [128, 128, 512], stage=2, block='0', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='1', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='2', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='3', **block_config)

    x = conv_block(x, 3, [256, 256, 1024], stage=3, block='0', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='1', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='2', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='3', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='4', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='5', **block_config)

    x = conv_block(x, 3, [512, 512, 2048], stage=4, block='0', **block_config)
    x = identity_block(x, 3, [512, 512, 2048], stage=4, block='1', **block_config)
    x = identity_block(x, 3, [512, 512, 2048], stage=4, block='2', **block_config)

    x = layers.GlobalAveragePooling2D()(x)
#     Final projection 
    x = layers.Dense(
      1024,
      activation="relu",
      name="fc")(
          x)

    # Create model.
    return models.Model(img_input, x, name='trunkmodel')

import torch

def init_weights_from_torch(tfmodel, torchpath):
    
    weights = torch.load(torchpath, map_location="cpu")["state_dict"]
    for ly in tfmodel.layers:
        # Conv layer
        prefix = "module.0."
        if len(ly.weights)==1:
            torchname = ly.weights[0].name.split("/")[0]
            # Double check if 2,3 or 3,2
            weight = weights[prefix+torchname+".weight"].cpu().numpy().transpose((2, 3, 1, 0))
            weights.pop(prefix+torchname+".weight")
            ly.weights[0].assign(weight)
        # BN
        elif len(ly.weights)==4:
            torchname = ly.weights[0].name.split("/")[0]
            weight = weights[prefix+torchname+".weight"].cpu().numpy()
            ly.weights[0].assign(weight)
            bias = weights[prefix+torchname+".bias"].cpu().numpy()
            ly.weights[1].assign(bias)
            running_mean = weights[prefix+torchname+".running_mean"].cpu().numpy()
            ly.weights[2].assign(running_mean)
            running_var = weights[prefix+torchname+".running_var"].cpu().numpy()
            ly.weights[3].assign(running_var)
            weights.pop(prefix+torchname+".weight")
            weights.pop(prefix+torchname+".bias")
            weights.pop(prefix+torchname+".running_mean")
            weights.pop(prefix+torchname+".running_var")
        # fc
        elif len(ly.weights)==2:
            torchname = ly.weights[0].name.split("/")[0]
            weight = weights[prefix+torchname+".weight"].cpu().numpy().T
            ly.weights[0].assign(weight)
            bias = weights[prefix+torchname+".bias"].cpu().numpy()
            ly.weights[1].assign(bias)
            weights.pop(prefix+torchname+".weight")
            weights.pop(prefix+torchname+".bias")
        else:
            assert len(ly.weights)==0


