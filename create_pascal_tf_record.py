#!/usr/bin/python
#!-*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import readfile
import numpy as np
import SimpleITK as sitk


def write_and_encode( data_list ):
    writer = tf.python_io.TFRecordWriter('../data/tfrecord/test.tfrecords')
    num=0
    for file_dir in data_list:
        file_dir=file_dir.split(',')
        Flair=readfile.load_itk(file_dir[0].strip()[1:-1])
        T1c=readfile.load_itk(file_dir[1].strip()[1:-1])
        T1= readfile.load_itk(file_dir[2].strip()[1:-1])
        T2 = readfile.load_itk(file_dir[3].strip()[1:-1])
        Mask = readfile.load_itk(file_dir[4].strip()[1:-1])
        for f,t1c,t1,t2,mask in zip(Flair,T1c,T1,T2,Mask):
           # temp=np.sum(f)+np.sum(t1c)+np.sum(t1)+np.sum(t2)

            #if(temp!=0):
                # temp=np.array(f)
                # f_raw=np.array(np.cast(np.array(f),np.int64)).tobytes()
                # t1c_raw=np.cast(np.array(t1c),np.int32).tobytes()
                # t1_raw=np.cast(np.array(t1),np.int32).tobytes()
                # t2_raw=np.cast(np.array(t2),np.int32).tobytes()
                # mask_raw=np.cast(np.array(mask),np.int32).tobytes()
            h, w = np.shape(f)
            f_raw=np.array(np.reshape(f,(h,w))+1000).tostring()

            t1c_raw=np.array(np.reshape(t1c,(h,w))+1000).tostring()
            t1_raw=np.array(np.reshape(t1,(h,w))+1000).tostring()
            t2_raw=np.array(np.reshape(t2,(h,w))+1000).tostring()
            mask_raw=np.array(np.reshape(mask,(h,w))+1000).tostring()

           # print(f_raw
            print(h,w)
            example=tf.train.Example(features=tf.train.Features(feature={
                "Flair": tf.train.Feature(bytes_list=tf.train.BytesList(value=[f_raw])),
                'T1c': tf.train.Feature(bytes_list=tf.train.BytesList(value=[t1c_raw])),
                'T1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[t1_raw])),
                'T2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[t2_raw])),
                'Mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_raw])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),

            }))
            num+=1
            writer.write(example.SerializeToString())
    print('%d image was writed to record file!'%num)
    writer.close()



def read_and_decode(filename):
    img_queue=tf.train.string_input_producer([filename])

    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(img_queue)
    features=tf.parse_single_example(serialized_example,
                                     features={
                                         'Flair':tf.FixedLenFeature([],tf.string),
                                         'T1c':tf.FixedLenFeature([],tf.string),
                                         'T1': tf.FixedLenFeature([], tf.string),
                                         'T2': tf.FixedLenFeature([], tf.string),
                                         'Mask': tf.FixedLenFeature([], tf.string),
                                         'height': tf.FixedLenFeature([], tf.int64),
                                         'width': tf.FixedLenFeature([], tf.int64),
                                     }
                                     )
    Flair_img=tf.decode_raw(features['Flair'],tf.int16)
    T1c_img=tf.decode_raw(features['T1c'],tf.int16)
    T1_img=tf.decode_raw(features['T1'],tf.int16)
    T2_img=tf.decode_raw(features['T2'],tf.int16)
    mask_img=tf.decode_raw(features['Mask'],tf.int16)
    height_img=features['height']
    width_img=features['width']
    h=240
    w=240
    Flair_img=tf.reshape(Flair_img,[h,w])
    T1c_img=tf.reshape(T1c_img,[h,w])
    T1_img=tf.reshape(T1_img,[h,w])
    T2_img=tf.reshape(T2_img,[h,w])
    mask_img=tf.reshape(mask_img,[h,w])

    input_tensor=tf.stack([Flair_img,T1c_img,T1_img,T2_img],-1)

    return input_tensor,mask_img

def net_inputs(batch_size, train_filename):

  resize_width=240
  resize_height=240

  image,mask= read_and_decode(train_filename)
  print(image)
  mask=tf.expand_dims(mask,-1)
  print(mask)

  image = tf.cast(tf.image.resize_images(image, [resize_height, resize_width]), tf.float32)

  mask = tf.cast(tf.image.resize_images(mask, [resize_height, resize_width]), tf.float32)

  image = image - 1000

  mask = mask - 1000  #mask = tf.squeeze(mask, squeeze_dims=[3])
  mask=tf.image.resize_images(mask,[resize_height,resize_width])


  images_batch, labels_batch= tf.train.batch(
      [image, mask],
      batch_size=batch_size,
      capacity=1)
  return images_batch, labels_batch


with open('./test_45.txt') as f:
    write_and_encode(f.readlines())