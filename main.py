#!/usr/bin/env python
# @_@ coding: utf-8 @_@
# Created  : Apr 11 18:34:58 2018
# Author   : Abhishake Kumar Bojja
# File     : main.py

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
######################################################################################################################
# Some parts of the code are adapted from:
# https://github.com/wayaai/SimGAN is publicly available under MIT License.
# https://github.com/carpedm20/simulated-unsupervised-tensorflow is publicly available under Apache License 2.0
# Assignment-4 of Introduction to Deep learning for Computer Vision course

# Data is taken from:
# https://www.kaggle.com/kmader/simgan-notebook/data
######################################################################################################################
import os
from config import get_config, print_usage
import tensorflow as tf
import h5py
import numpy as np
from train import *


def main(config):
    """The main function."""


    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    if not os.path.exists(config.fig_save_dir):
        os.makedirs(config.fig_save_dir)
    mynet = MyNetwork(config)


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)