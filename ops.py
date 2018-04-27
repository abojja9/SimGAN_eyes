#!/usr/bin/env python
# @_@ coding: utf-8 @_@
# Created  : Apr 12 13:53:44 2018
# Author   : Abhishek_Kumar_Bojja 
# File     : ops.py

# Description:
# Maintainer:
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C)

# Code:

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import h5py
from tensorflow.python.framework import ops

from utils import *
import time
from keras.layers import Add
from keras import applications
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image


def batch_norm(x, is_training, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=name
                                        )


def conv2d(input_, output_dim, ks=3, s=2, padding='SAME', name="conv2d"):
    return slim.conv2d(input_,
                       output_dim,
                       ks,
                       s,
                       padding=padding,
                       activation_fn=None,
                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                       biases_initializer=tf.zeros_initializer(),
                       scope=name
                       )


# def deconv2d(input_, output_dim, ks=4, s=2, padding='SAME', name="deconv2d"):
#     return slim.conv2d_transpose(input_,
#                                  output_dim,
#                                  ks,
#                                  s,
#                                  padding=padding,
#                                  activation_fn=None,
#                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
#                                  biases_initializer=None,
#                                  scope=name
#                                  )

def sparse_sce_loss(logits, labels, name="sparse_sce"):
    return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                         [1, 2],
                         name=name
                         )

def resnet_block(input, filters=64, kernel_size=3, i=1, is_training=True):
    """

    :param input: Convolved Input image tensor to ResNet block.
    :return: Output tensor from ResNet block.
    """
    input_identity = input

    # First Block
    res1 = conv2d(input, filters, kernel_size, s=1, name="res{}_1_conv2d".format(i))
    # res1 = batch_norm(res1, is_training)
    res1 = lrelu(res1, 0.0)

    # Second Block
    # res2 = conv2d(res1, filters, kernel_size, s=1)
    # res2 = batch_norm(res2, is_training)
    # res2 = lrelu(res2, 0.0)

    # Third Block
    res3 = conv2d(res1, filters, kernel_size, s=1, name="res{}_1_conv2d".format(i))
    # res3 = batch_norm(res3, is_training)

    output_identity = res3 + input_identity
    output = lrelu(output_identity, 0.0)

    return output

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x)

###########################################################################
# Below parts of the code are adapted from:
# https://github.com/wayaai/SimGAN is publicly available under MIT License.
###########################################################################

def get_image_batch(generator, batch_size):
    """keras generators may generate an incomplete batch for the last batch"""
    img_batch = generator.next()
    if len(img_batch) != batch_size:
        img_batch = generator.next()
    # print (img_batch.shape)
    # print(img_batch.dtype)
    # print("len:", len(img_batch))
    # print ("bsize:",batch_size)
    assert len(img_batch) == batch_size

    return img_batch

class ImageHistoryBuffer(object):
    def __init__(self, shape, max_size, batch_size):
        """
        Initialize the class's state.

        :param shape: Shape of the data to be stored in the image history buffer
                      (i.e. (0, img_height, img_width, img_channels)).
        :param max_size: Maximum number of images that can be stored in the image history buffer.
        :param batch_size: Batch size used to train GAN.
        """
        self.image_history_buffer = np.zeros(shape=shape)
        self.max_size = max_size
        self.batch_size = batch_size

    def add_to_image_history_buffer(self, images, nb_to_add=None):
        """
        To be called during training of GAN. By default add batch_size // 2 images to the image history buffer each
        time the generator generates a new batch of images.

        :param images: Array of images (usually a batch) to be added to the image history buffer.
        :param nb_to_add: The number of images from `images` to add to the image history buffer
                          (batch_size / 2 by default).
        """
        if not nb_to_add:
            nb_to_add = self.batch_size // 2

        if len(self.image_history_buffer) < self.max_size:
            np.append(self.image_history_buffer, images[:nb_to_add], axis=0)
        elif len(self.image_history_buffer) == self.max_size:
            self.image_history_buffer[:nb_to_add] = images[:nb_to_add]
        else:
            assert False

        np.random.shuffle(self.image_history_buffer)

    def get_from_image_history_buffer(self, nb_to_get=None):
        """
        Get a random sample of images from the history buffer.

        :param nb_to_get: Number of images to get from the image history buffer (batch_size / 2 by default).
        :return: A random sample of `nb_to_get` images from the image history buffer, or an empty np array if the image
                 history buffer is empty.
        """
        if not nb_to_get:
            nb_to_get = self.batch_size // 2

        try:
            return self.image_history_buffer[:nb_to_get]
        except IndexError:
            return np.zeros(shape=0)
