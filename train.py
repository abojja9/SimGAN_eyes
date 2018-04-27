#!/usr/bin/env python
# @_@ coding: utf-8 @_@
# Created  : Apr 11 18:37:15 2018
# Author   : Abhishek Kumar Bojja
# File     : train.py

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



import numpy as np
import tensorflow as tf
from tqdm import trange
import h5py
import os
from keras import applications
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image

from config import get_config, print_usage
from ops import *
from utils import *


class MyNetwork(object):
    """Network class """

    def __init__(self, config):
        """Initializer

        This function initializes the network and constructs all the
        computational workflow of our network.

        Parameters
        ----------

        config : Python namespace
            Configuration namespace.

        """

        self.config = config
        # Get shape
        self.image_width = config.image_width
        self.image_height = config.image_height
        self.image_channels = config.image_channels

        # Build the network
        self._build_placeholder()
        self._build_data()
        self._build_model()
        self._build_loss()
        self._build_optim()
        # self._build_eval()
        self._build_summary()
        self._build_writer()
        self.train()

    def _build_placeholder(self):
        """Build placeholders."""

        self.real_input = tf.placeholder(tf.float32,
                                         shape=(None, self.image_height, self.image_width, self.image_channels))
        self.synthetic_input = tf.placeholder(tf.float32,
                                              shape=(None, self.image_height, self.image_width, self.image_channels))
        self.refined_buffer_input = tf.placeholder(tf.float32,
                                    shape=(None, self.image_height, self.image_width, self.image_channels))
        # self.learning_rate = tf.placeholder(tf.float32, shape=())
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.is_history_buffer = tf.placeholder(tf.bool, name='is_history_buffer')




    def _build_data(self):
        """

        :return:
        """

        with h5py.File(self.config.synthetic_data, 'r') as syn_file:
            print('Number of synthetic images found:', len(syn_file['image']))
            self.syn_image_stack = np.stack([np.expand_dims(a, -1) for a in syn_file['image'].values()], 0)

        with h5py.File(self.config.real_data, 'r') as real_file:
            print('Number of real images found:', len(real_file['image']))
            self.real_image_stack = np.stack([np.expand_dims(a, -1) for a in real_file['image'].values()], 0)

        print(self.syn_image_stack.shape, self.syn_image_stack.dtype, self.syn_image_stack.min(), self.syn_image_stack.max())
        print(self.real_image_stack.shape, self.real_image_stack.dtype, self.real_image_stack.min() , self.real_image_stack.max())

        syn_data_gen_args = dict(preprocessing_function=applications.xception.preprocess_input,
                                 data_format='channels_last',
                                )
        real_data_gen_args = dict(rotation_range=0.,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  zoom_range=0.1,
                                  horizontal_flip=True,
                                  preprocessing_function=applications.xception.preprocess_input,
                                  data_format='channels_last',
                                 )
        syn_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**syn_data_gen_args)
        real_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**syn_data_gen_args) #real_data_gen_args) #
        self.synthetic_generator = syn_datagen.flow(x=self.syn_image_stack,
                                                    batch_size=self.config.batch_size
                                                    )
        self.real_generator = real_datagen.flow(x=self.real_image_stack,
                                                batch_size=self.config.batch_size
                                                )



    def _build_model(self):

        self.global_step = tf.Variable(initial_value=0, dtype=tf.int32, name='global_step')




    def _build_loss(self):
        """

        :return:
        """

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):

            self.refined_image = self._build_refiner(self.synthetic_input)


            # General GAN LOSS
            print (self.config.loss_fn)

            if self.config.loss_fn == 'GAN':
                self.real_image = self._build_discriminator(self.real_input)[0]
                self.fake_image = self._build_discriminator(self.refined_buffer_input, reuse=True)[0]
                disc_loss_real = tf.reduce_mean(sparse_sce_loss(logits=self.real_image,
                                                 labels=tf.zeros_like(self.real_image, dtype=tf.int32)[:,:,:,0],
                                                )
                )

                disc_loss_fake = tf.reduce_mean(sparse_sce_loss(logits=self.fake_image,
                                                 labels=tf.ones_like(self.fake_image, dtype=tf.int32)[:,:,:,0],
                                                )
                )

                self.disc_loss_adversarial = tf.reduce_mean(disc_loss_real + disc_loss_fake,
                                                            name="discriminator_loss"
                                                            )

                # self.disc_loss_adversarial = discriminator_loss( self.real_image,  self.fake_image)
                # Refiner Loss
                self.refiner_loss_adversarial = tf.reduce_mean(sparse_sce_loss(logits=self.fake_image,
                                                            labels=tf.zeros_like(self.fake_image, dtype=tf.int32)[:,:,:,0],
                                                            name='refiner_loss_adversarial'
                                                            )
                                                        )
                self.regularization_loss = self.config.reg_lambda * tf.reduce_mean(tf.abs(self.refined_image - self.synthetic_input),name='regularization_loss')

                self.refiner_loss = tf.reduce_mean(self.refiner_loss_adversarial + self.regularization_loss,
                                                   name='refiner_loss'
                                                   )


            ## SimGAN Loss
            elif self.config.loss_fn == 'SimGAN':
                print(self.config.loss_fn)
                self.real_image = self._build_discriminator(self.real_input)[1]
                self.fake_image = self._build_discriminator(self.refined_image, reuse=True)[1]
                self.fake_image_hist = self._build_discriminator(self.refined_buffer_input, reuse=True)[1]
                # self.real_image = tf.reshape(self.real_image, (self.config.batch_size, -1))
                # self.fake_image = tf.reshape(self.fake_image, (self.config.batch_size, -1))

                disc_loss_real = -tf.reduce_mean(tf.log(tf.clip_by_value(self.real_image, self.config.epsilon, 1.0 - self.config.epsilon)))

                # tf.clip_by_value(1-self.real_image, -(1 + self.config.epsilon), 1 + self.config.epsilon)
                disc_loss_fake = -tf.reduce_mean(tf.log(tf.clip_by_value(1.0-self.fake_image, self.config.epsilon, 1.0 - self.config.epsilon)))
                disc_loss_fake_hist = -tf.reduce_mean(tf.log(tf.clip_by_value(1.0-self.fake_image_hist, self.config.epsilon, 1.0 - self.config.epsilon)))

                self.disc_loss_adversarial = tf.reduce_mean(disc_loss_fake + disc_loss_fake_hist) + disc_loss_real

                self.regularization_loss = tf.reduce_mean(tf.abs(self.refined_image - self.synthetic_input), name='regularization_loss')


                self.refiner_loss_adversarial = -tf.reduce_mean(
                    tf.log(tf.clip_by_value(self.fake_image, self.config.epsilon, 1.0 - self.config.epsilon)),
                    name='refiner_loss_adversarial')

                self.refiner_loss = tf.reduce_mean(self.refiner_loss_adversarial + self.config.reg_lambda * self.regularization_loss,
                                                       name='refiner_loss'
                                                     )






            # Record summary for loss
            tf.summary.scalar("refiner loss", self.refiner_loss)
            tf.summary.scalar("refiner adv loss", self.refiner_loss_adversarial)
            tf.summary.scalar("regularization loss", self.regularization_loss)
            tf.summary.scalar("discriminator loss", self.disc_loss_adversarial)

            # Record summary for Images
            tf.summary.image("synthetic image", self.synthetic_input)
            tf.summary.image("unlabelled real image", self.real_input)
            tf.summary.image("refined image", self.refined_image)





    def _build_optim(self):
        """Build optimizer related ops and vars."""

        with tf.variable_scope("Optim", reuse=tf.AUTO_REUSE):
            t_vars = tf.trainable_variables()
            dis_vars = [var for var in t_vars if "Discriminator" in var.name]
            refiner_vars = [var for var in t_vars if "Refiner" in var.name]

            # print(dis_vars)
            print(refiner_vars)

            # self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in dis_vars]

            # Check for which optimizer to use
            if self.config.optim == 'adam':
                optimizer = tf.train.AdamOptimizer
                optim_args = (self.config.learning_rate, self.config.beta1)
            elif self.config.optim == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer
                optim_args = (self.config.learning_rate,)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                print (optimizer)
                self.discriminator_opt = optimizer(*optim_args).minimize(
                    self.disc_loss_adversarial,
                    var_list=dis_vars,
                    global_step=self.global_step
                )
                self.refiner_opt = optimizer(*optim_args).minimize(
                    self.refiner_loss,
                    var_list=refiner_vars,
                    global_step=self.global_step
                )
                self.regularize_refiner_opt = optimizer(*optim_args).minimize(
                    self.regularization_loss,
                    var_list=refiner_vars,
                    global_step=self.global_step
                )




    def _build_writer(self):
        """Build the writers and savers"""
        self.summary_train = tf.summary.FileWriter(logdir=self.config.log_dir + "/train")
        # self.summary_discriminator = tf.summary.FileWriter(logdir=self.config.log_dir + "/train/discriminator")
        # self.summary_va = tf.summary.FileWriter(logdir=self.config.log_dir + "/valid")

        # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()

        self.save_file_cur = self.config.log_dir + "/model"

        # Saving the best model
        self.save_file_best = self.config.save_dir + "/model"

        # Saving images during training
        self.save_figure = self.config.fig_save_dir


    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()


    def train(self):
        """

        :param self:
        :return:
        """


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iterator = sess.run(self.global_step)

            image_history_buffer = ImageHistoryBuffer((0, self.image_height, self.image_width, self.image_channels),
                                                      self.config.batch_size * 100,
                                                      self.config.batch_size)

            # Pre training the refiner and discriminator
            for step in range(200):
                # print("init trianing {}".format(step))
                for r in range(5):
                    syn_image_batch = get_image_batch(self.synthetic_generator, self.config.batch_size)
                    real_image_batch = get_image_batch(self.real_generator, self.config.batch_size)
                    refined_image_batch = sess.run(self.refined_image,
                                                   feed_dict={self.synthetic_input: syn_image_batch
                                                              })
                    fetch = {"optimizer": self.regularize_refiner_opt,
                             "loss": self.regularization_loss,
                             "summary": self.summary_op,
                             "global_step": self.global_step}

                    feed_dict = {self.synthetic_input:syn_image_batch,
                                 self.refined_buffer_input: refined_image_batch,
                                 self.real_input: real_image_batch,
                                 self.is_training: True,
                                 self.is_history_buffer: False,
                                 }
                    res_ref = sess.run(fetch, feed_dict=feed_dict)
                #
                    self.summary_train.add_summary(res_ref["summary"], global_step=res_ref["global_step"])
                #
                if step % self.config.print_every == 0:
                    print ("In pre training refiner loss at step:{} is: {}".format(step, res_ref["loss"]))
            #
                self.summary_train.flush()

                for d in range(1):

                    real_image_batch = get_image_batch(self.real_generator, self.config.batch_size)
                    syn_image_batch = get_image_batch(self.synthetic_generator, self.config.batch_size)
                    refined_image_batch = sess.run(self.refined_image,
                                                   feed_dict={self.synthetic_input: syn_image_batch
                                                              })

                    # use a history of refined images
                    half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
                    image_history_buffer.add_to_image_history_buffer(refined_image_batch)

                    if len(half_batch_from_image_history):
                        refined_image_batch[:self.config.batch_size // 2] = half_batch_from_image_history

                    fetch = {
                             "optimizer": self.discriminator_opt,
                             "loss": self.disc_loss_adversarial,
                             "summary": self.summary_op,
                             "global_step": self.global_step}

                    feed_dict = {self.synthetic_input: syn_image_batch,
                                 self.real_input: real_image_batch,
                                 self.refined_buffer_input: refined_image_batch,
                                 self.is_training: True,
                                 self.is_history_buffer: True,

                                 }
                    res_dis = sess.run(fetch, feed_dict=feed_dict)
                    self.summary_train.add_summary(res_dis["summary"], global_step=res_dis["global_step"])

                if step % self.config.print_every == 0:
                    print ("In pre training discriminator loss at step:{} is {}".format(step, res_dis["loss"]))

                self.summary_train.flush()
                self.saver_cur.save(sess=sess, save_path=self.save_file_cur,
                                    write_meta_graph=False, global_step=res_ref["global_step"])

            for epoch_i in range(self.config.num_epochs):
                num_batches = self.syn_image_stack.shape[0]//self.config.batch_size
                for batch in range(num_batches):
                    # print("tarining refiner")
                    for step in range(self.config.k_ref * 1):
                        syn_image_batch = get_image_batch(self.synthetic_generator, self.config.batch_size)
                        real_image_batch = get_image_batch(self.real_generator, self.config.batch_size)
                        refined_image_batch = sess.run(self.refined_image,
                                                       feed_dict={self.synthetic_input: syn_image_batch
                                                                  })

                        fetch = {"optimizer": self.refiner_opt,
                                 "loss": self.refiner_loss,
                                 "summary": self.summary_op,
                                 "global_step": self.global_step
                                 }

                        feed_dict = {self.synthetic_input: syn_image_batch,
                                     self.real_input: real_image_batch,
                                     self.refined_buffer_input: refined_image_batch,
                                     self.is_training: True,
                                     self.is_history_buffer: False,

                                     }
                        res_ref = sess.run(fetch, feed_dict=feed_dict)

                        self.summary_train.add_summary(res_ref["summary"], global_step=res_ref["global_step"])

                    if iterator % self.config.print_every == 0:
                        print("Training refiner loss at iteration:{} is: {}".format(iterator, res_ref["loss"]))
                    if iterator % self.config.save_every == 0:
                        figure_name = 'refined_image_batch_train_step_{}.png'.format(iterator)
                        print('Saving batch of refined images during training at iteration: {}.'.format(iterator))
                        syn_image_batch = get_image_batch(self.synthetic_generator, self.config.batch_size)

                        refined_image_batch = sess.run(self.refined_image,
                                                       feed_dict={self.synthetic_input: syn_image_batch
                                                                  })
                        # synthetic_image_batch = get_image_batch(synthetic_generator)
                        plot_batch(np.concatenate((syn_image_batch, refined_image_batch)),
                        self.save_figure + "/" + figure_name,
                        label_batch=['Synthetic'] * self.config.batch_size + ['Refined'] * self.config.batch_size
                        )

                        images_grid = images_square_grid(refined_image_batch, "L")
                        plt.imshow(images_grid, cmap='bone')
                        plt.savefig(os.path.join(self.save_figure + "/" + "{}_refined_image".format(iterator)))
                        plt.close()

                    self.summary_train.flush()

                    # print("tarining discriminator")
                    for step in range(self.config.k_dis * 10):

                        real_image_batch = get_image_batch(self.real_generator, self.config.batch_size)
                        syn_image_batch = get_image_batch(self.synthetic_generator, self.config.batch_size)
                        refined_image_batch = sess.run(self.refined_image,
                                                       feed_dict={self.synthetic_input: syn_image_batch
                                                                  })

                        # use a history of refined images
                        half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
                        image_history_buffer.add_to_image_history_buffer(refined_image_batch)

                        if len(half_batch_from_image_history):
                            refined_image_batch[:self.config.batch_size // 2] = half_batch_from_image_history

                        fetch = {
                                 "optimizer": self.discriminator_opt,
                                 "loss": self.disc_loss_adversarial,
                                 "summary": self.summary_op,
                                 "global_step": self.global_step}

                        feed_dict = {self.synthetic_input: syn_image_batch,
                                     self.real_input: real_image_batch,
                                     self.refined_buffer_input: refined_image_batch,
                                     self.is_training: True,
                                     self.is_history_buffer: True,

                                     }
                        res_dis = sess.run(fetch, feed_dict=feed_dict)
                        self.summary_train.add_summary(res_dis["summary"], global_step=res_dis["global_step"])

                    if iterator % self.config.print_every == 0:
                        print("Training discriminator loss at iteration:{} is {}".format(iterator, res_dis["loss"]))
                    self.summary_train.flush()
                    self.saver_cur.save(sess=sess, save_path=self.save_file_cur,
                                        write_meta_graph=False, global_step=res_ref["global_step"])


                    iterator += 1


    def _build_refiner(self, refiner_input, reuse=False):
        """
        :param refiner_input: Input tensor containing synthetic image to the Refiner
        :param reuse:
        :return: Output tensor containing refined image
        :return:
        """
        with tf.variable_scope("Refiner", reuse=reuse):
        #TODO
            x = conv2d(refiner_input, 64, 3, 1, name="conv_2d_1"); print (x.shape)
            x = lrelu(x, 0.0);print (x.shape)
            for i in range(self.config.num_resnet_blocks):
                x = resnet_block(x, 64, 3, i, self.is_training);print (x.shape)

            x = conv2d(x, self.image_channels, 1, 1, name="conv2d_{}".format(i+1)) ;print (x.shape)# Output of resnet block is passed to 1 x 1 conv2d layer
            x = tf.nn.tanh(x);print (x.shape)
            return x

    def _build_discriminator(self, disc_input, reuse=False):
        """
              The discriminator network, layers are as follows:
               (1) Conv3x3, stride=2, feature maps=96, -> relu
               (2) Conv3x3,stride=2, feature maps=64,  -> relu
               (3) MaxPool3x3, stride=1,
               (4) Conv3x3, stride=1, feature maps=32, -> relu
               (5) Conv1x1, stride=1, feature maps=32, -> relu
               (6) Conv1x1, stride=1, feature maps=2, -> relu
               (7) Softmax.
              :param disc_input: Input tensor to the discriminator
              :param reuse:
              :return: Output tensor containing feature maps for patches of the input image
              """
        with tf.variable_scope("Discriminator", reuse=reuse):
            x = conv2d(disc_input, 96, 3, 2, name="conv_2d_1")
            x = lrelu(x, 0.0)
            x = conv2d(x, 64, 3, 2, name="conv_2d_2")
            x = lrelu(x, 0.0)
            x = tf.layers.max_pooling2d(x, 3, 1, padding='same')
            x = conv2d(x, 32, 3, 1, name="conv_2d_3")
            x = lrelu(x, 0.0)
            x = conv2d(x, 32, 1, 1, name="conv_2d_4")
            x = lrelu(x, 0.0)
            logits = conv2d(x, 1, 1, 1, name="conv_2d_5")
            output = tf.nn.sigmoid(logits)
            # output = tf.reshape(output, (-1))
            return logits, output
















































