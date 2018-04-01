#!/usr/bin/python
#!-*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np


def read_and_decode_single_example(filename, islable=False):
 # print filename
  filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(filename,tf.string), shuffle=False)

  reader=tf.WholeFileReader()
  _, image_file = reader.read(filename_queue)

  image = tf.cast(tf.image.decode_png(image_file, channels=1), tf.float32)

  #image.set_shape([959,776, 3])
  #image = tf.pad(image, [[0, 0], [78, 78], [0, 0]], mode='CONSTANT')
  if islable:
   # image=tf.cast(tf.image.rgb_to_grayscale(image)//63,tf.int32)
    image = tf.cast(tf.image.rgb_to_grayscale(image), tf.int32)
    #image=tf.one_hot(tf.cast(image,tf.int32),depth=2,axis =-1)
    #image=image
  else:
    #image /= 255
    image=tf.image.rgb_to_grayscale(image)

  return image


def net_inputs(batch_size, Flair_filename,T1c_filename,T1_filename,T2_filename, train_labels_filename):

  resize_width=224
  resize_height=224
  central_fraction=0.8

  Flair_image = read_and_decode_single_example(Flair_filename)
  #Flair_image = tf.image.central_crop(Flair_image, central_fraction=central_fraction)
  Flair_image=tf.image.resize_images(Flair_image,[resize_height,resize_width])

  #Flair_image=tf.expand_dims(Flair_image,2)

  T1c_image = read_and_decode_single_example(T1c_filename)
  #T1c_image=tf.image.central_crop(T1c_image,central_fraction=central_fraction)
  T1c_image=tf.image.resize_images(T1c_image,[resize_height,resize_width])

 # T1c_image=tf.expand_dims(T1c_image,2)

  T1_image = read_and_decode_single_example(T1_filename)
  #T1_image = tf.image.central_crop(T1_image, central_fraction=central_fraction)
  T1_image = tf.image.resize_images(T1_image, [resize_height, resize_width])

  #T1_image=tf.expand_dims(T1_image,2)

  T2_image = read_and_decode_single_example(T2_filename)
  #T2_image = tf.image.central_crop(T2_image,central_fraction=central_fraction)
  T2_image = tf.image.resize_images(T2_image, [resize_height, resize_width])

  #T2_image=tf.expand_dims(T2_image,2)

  label = read_and_decode_single_example(train_labels_filename,islable=True)
  label = tf.image.central_crop(label, central_fraction=central_fraction)
  label=tf.expand_dims(label,[0])
  label = tf.image.resize_nearest_neighbor(label,[resize_height,resize_width])
  label=tf.squeeze(label,[0])


  Flair_image_batch, T1c_image_batch, T1_image_batch, T2_image_batch, labels_batch= tf.train.batch(
    [Flair_image,T1c_image,T1_image,T2_image, label],
    batch_size=batch_size,
    capacity=2000)
 # print Flair_image
  images_batch=tf.concat([Flair_image_batch,T1c_image_batch,T1_image_batch,T2_image_batch],3)
  return images_batch,labels_batch

  # else:
  #   return tf.train.batch([Flair_image,T1c_image,T1_image,T2_image ], batch_size=batch_size, capacity=2000)

def test_inputs(batch_size, Flair_filename,T1c_filename,T1_filename,T2_filename):

  resize_width=224
  resize_height=224
  central_fraction=0.8

  Flair_image = read_and_decode_single_example(Flair_filename)
  #Flair_image = tf.image.central_crop(Flair_image, central_fraction=central_fraction)
  Flair_image=tf.image.resize_images(Flair_image,[resize_height,resize_width])

  #Flair_image=tf.expand_dims(Flair_image,2)

  T1c_image = read_and_decode_single_example(T1c_filename)
  #T1c_image=tf.image.central_crop(T1c_image,central_fraction=central_fraction)
  T1c_image=tf.image.resize_images(T1c_image,[resize_height,resize_width])

 # T1c_image=tf.expand_dims(T1c_image,2)

  T1_image = read_and_decode_single_example(T1_filename)
 # T1_image = tf.image.central_crop(T1_image, central_fraction=central_fraction)
  T1_image = tf.image.resize_images(T1_image, [resize_height, resize_width])

  #T1_image=tf.expand_dims(T1_image,2)

  T2_image = read_and_decode_single_example(T2_filename)
  #T2_image = tf.image.central_crop(T2_image,central_fraction=central_fraction)
  T2_image = tf.image.resize_images(T2_image, [resize_height, resize_width])

  #T2_image=tf.expand_dims(T2_image,2)



  Flair_image_batch, T1c_image_batch, T1_image_batch, T2_image_batch= tf.train.batch(
    [Flair_image,T1c_image,T1_image,T2_image],
    batch_size=batch_size,
    capacity=2000)
 # print Flair_image
  images_batch=tf.concat([Flair_image_batch,T1c_image_batch,T1_image_batch,T2_image_batch],3)
  return images_batch



def read_file_list(file_path):

    train_object=[]

    with open(file_path) as f:

      for i in f.readlines():
        temp=i.strip('\n').split(',')
        train_object.append(temp)


    return np.transpose(train_object)



