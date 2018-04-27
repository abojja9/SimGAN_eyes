#!/usr/bin/env python
# @_@ coding: utf-8 @_@
# Created  : Apr 12 13:52:28 2018
# Author   : Abhishek_Kumar_Bojja 
# File     : utils.py

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
###########################################################################
# Some parts of the code are adapted from:
# https://github.com/wayaai/SimGAN is publicly available under MIT License.
###########################################################################

import matplotlib
from matplotlib import pyplot as plt
import os
import numpy as np
from itertools import groupby
from skimage.util.montage import montage2d
from PIL import Image
import math


def plot_batch(image_batch, figure_path, label_batch=None):
    all_groups = {label: montage2d(np.stack([img[:, :, 0] for img, lab in img_lab_list], 0))
                  for label, img_lab_list in groupby(zip(image_batch, label_batch), lambda x: x[1])}
    print (len(all_groups))
    fig, c_axs = plt.subplots(1, len(all_groups), figsize=(len(all_groups) * 2, 2), dpi=300)
    for c_ax, (c_label, c_mtg) in zip(c_axs, all_groups.items()):
        c_ax.imshow(c_mtg, cmap='bone')
        c_ax.set_title(c_label)
        c_ax.axis('off')
    fig.savefig(os.path.join(figure_path))
    print ("image_saved")
    plt.close()

def images_square_grid(images, mode):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :param mode: The mode to use for images
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    # Scale to 0-255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # Put images in a square arrangement
    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))
    if mode == 'L':
        images_in_square = np.squeeze(images_in_square, 4)

    # Combine images to grid image
    new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, mode)
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im

# def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
#     """
#     Show example output for the generator
#     :param sess: TensorFlow session
#     :param n_images: Number of Images to display
#     :param input_z: Input Z Tensor
#     :param out_channel_dim: The number of channels in the output image
#     :param image_mode: The mode to use for images ("RGB" or "L")
#     """
#     cmap = None if image_mode == 'RGB' else 'gray'
#     z_dim = input_z.get_shape().as_list()[-1]
#     example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])
#
#     samples = sess.run(
#         generator(input_z, out_channel_dim, False),
#         feed_dict={input_z: example_z})
#
#     images_grid = helper.images_square_grid(samples, image_mode)
#     pyplot.imshow(images_grid, cmap=cmap)
#     pyplot.show()