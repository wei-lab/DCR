#!/usr/bin/python
#!-*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import readfile
import numpy as np


def read_and_decode( filename ):
    img_queue = tf.train.string_input_producer( [filename] )

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read( img_queue )
    features = tf.parse_single_example( serialized_example,
                features = {
                    'Flair' : tf.FixedLenFeature( [], tf.string ), 
                      'T1c' : tf.FixedLenFeature( [], tf.string ), 
                       'T1' : tf.FixedLenFeature( [], tf.string ), 
                       'T2' : tf.FixedLenFeature( [], tf.string ), 
                     'Mask' : tf.FixedLenFeature( [], tf.string ), 
                   'height' : tf.FixedLenFeature( [], tf.int64 ), 
                    'width' : tf.FixedLenFeature( [], tf.int64 ), 
                })
    Flair_img = tf.decode_raw( features['Flair'], tf.int16 )
    T1c_img = tf.decode_raw( features['T1c'], tf.int16 )
    T1_img = tf.decode_raw( features['T1'], tf.int16 )
    T2_img = tf.decode_raw( features['T2'], tf.int16 )
    mask_img = tf.decode_raw( features['Mask'], tf.int16 )
    height_img = features['height']
    width_img = features['width']
    h = 240
    w = 240
    Flair_img = tf.reshape( Flair_img, [h, w] )
    T1c_img = tf.reshape( T1c_img, [h, w] )
    T1_img = tf.reshape( T1_img, [h, w] )
    T2_img = tf.reshape( T2_img, [h, w] )
    mask_img = tf.reshape( mask_img, [h,w] )

    input_tensor = tf.stack( [Flair_img, T1c_img, T1_img, T2_img], -1 )

    return input_tensor, mask_img


def net_inputs( batch_size, train_filename ):
  resize_width = 224
  resize_height = 224

  image,mask = read_and_decode( train_filename )
  mask = tf.expand_dims( mask, -1 )

  image = tf.cast( tf.image.resize_images( image, [resize_height, resize_width] ), tf.float32 )

  mask = tf.cast( tf.image.resize_images(mask, [resize_height, resize_width]), tf.float32 )

  image = image - 1000
  mask = mask - 1000  
  mask = tf.image.resize_images( mask, [resize_height, resize_width] )

  images_batch, labels_batch = tf.train.batch( [image, mask], 
                                    batch_size = batch_size,
                                    capacity = 1 )
  return images_batch, labels_batch


def read_and_decode_test( filename ):
    img_queue = tf.train.string_input_producer( [filename] )

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read( img_queue )
    features = tf.parse_single_example( serialized_example, 
                  features = {
                       'Flair' : tf.FixedLenFeature( [], tf.string ),
                         'T1c' : tf.FixedLenFeature( [], tf.string ),
                          'T1' : tf.FixedLenFeature( [], tf.string ),
                          'T2' : tf.FixedLenFeature( [], tf.string ),
                      'height' : tf.FixedLenFeature( [], tf.int64 ),
                       'width' : tf.FixedLenFeature( [], tf.int64 ),
                  })
    Flair_img = tf.decode_raw( features['Flair'], tf.int16 )
    T1c_img = tf.decode_raw( features['T1c'], tf.int16 )
    T1_img = tf.decode_raw( features['T1'], tf.int16 )
    T2_img = tf.decode_raw( features['T2'], tf.int16 )
    height_img = features['height']
    width_img = features['width']
    h = 240
    w = 240
    Flair_img = tf.reshape( Flair_img, [h,w] )
    T1c_img = tf.reshape( T1c_img, [h,w] )
    T1_img = tf.reshape( T1_img, [h,w] )
    T2_img = tf.reshape( T2_img, [h,w] )

    input_tensor = tf.stack( [Flair_img, T1c_img, T1_img, T2_img], -1 )

    return input_tensor


def net_inputs_test( batch_size, test_filename ):
  resize_width = 224
  resize_height = 224

  image =  read_and_decode_test( test_filename )

  image  =  tf.cast( tf.image.resize_images(image, [resize_height, resize_width]), tf.float32 )

  image  =  image - 1000
  images_batch =  tf.train.batch( [image], batch_size = batch_size, capacity = 2000)

  return images_batch
