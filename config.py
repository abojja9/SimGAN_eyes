#!/usr/bin/env python
# @_@ coding: utf-8 @_@
# Created  : Apr 11 18:28:51 2018
# Author   : Abhishek Kumar Bojja
# File     : config.py

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
##########################################################################
# Some parts of the code are adapted from:

# Assignment-4 of Introduction to Deep learning for Computer Vision course
##########################################################################

import argparse


# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for argparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg



# Data
data_arg = add_argument_group('Data')

data_arg.add_argument('--image_height', type=int, default=35)
data_arg.add_argument('--image_width', type=int, default=55)
data_arg.add_argument('--image_channels', type=int, default=1)


# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")


train_arg.add_argument("--data_dir", type=str,
                       default="/media/abojja/HD/Abhi/dl_project/data/",
                       help="Directory with real and synthetic eyes")
train_arg.add_argument("--synthetic_data", type=str,
                       default="/media/abojja/HD/Abhi/dl_project/data/synthetic_gaze.h5",
                       help="Directory with synthetic eyes")
train_arg.add_argument("--real_data", type=str,
                       default="/media/abojja/HD/Abhi/dl_project/data/real_gaze.h5",
                       help="Directory with real eyes")
train_arg.add_argument("--learning_rate", type=float,
                       default=1e-4,
                       help="Learning rate (gradient step size)")
train_arg.add_argument("--beta1", type=float,
                       default=0.5,
                       help="beta1 for adam")

train_arg.add_argument("--batch_size", type=int,
                       default=32,
                       help="Size of each training batch")
train_arg.add_argument("--pre_train_steps_refiner", type=int,
                       default=1000,
                       help="Training the refiner before actual training")
train_arg.add_argument("--pre_train_steps_disc", type=int,
                       default=200,
                       help="Training the discriminator before actual training")
train_arg.add_argument("--k_ref", type=int,
                       default=1,
                       help="Training the discriminator before actual training")
train_arg.add_argument("--k_dis", type=int,
                       default=1,
                       help="Training the discriminator before actual training")
train_arg.add_argument("--print_every", type=int,
                       default=100,
                       help="Display output for every * steps")
train_arg.add_argument("--save_every", type=int,
                       default=1000,
                       help="Display output for every * steps")
train_arg.add_argument("--epsilon", type=int,
                       default=1e-10,
                       help="epsilon value for log loss")
train_arg.add_argument("--loss_fn", type=str,
                       default='SimGAN',
                       choices=["GAN", "SimGAN"],
                       help="loss function to use")
train_arg.add_argument("--optim", type=str,
                       default='adam',
                       choices=["sgd", "adam"],
                       help="Optimizer to use")

train_arg.add_argument("--num_epochs", type=int,
                       default=100,
                       help="Number of epochs to train")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save the best model")
train_arg.add_argument("--fig_save_dir", type=str,
                       default="./figures",
                       help="Directory to save images during training")


# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")


model_arg.add_argument("--reg_lambda", type=float,
                       default=1e-4,
                       help="Regularization strength")
model_arg.add_argument("--num_resnet_blocks", type=int,
                       default=4,
                       help="Number of resnet blocks ")




def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
