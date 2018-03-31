#!/usr/bin/python
#!-*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  tensorflow as tf
from  PIL import Image
import numpy as np
import scipy.misc

# label_colours = [(0,0,0)
#                 # 0=background
#                 ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
#                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
#                 ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
#                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
#                 ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
#                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
#                 ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
#                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

label_colours = [(0,0,0),
                # 0=background
                (255,222,173),(187,255,255),(100,149,237)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(255,127,80),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/mo

def prepare_label(input_batch,new_size,num_classes,one_hot=True):

    with tf.name_scope('label_encode'):
        input_batch=tf.image.resize_nearest_neighbor(input_batch,new_size)
        input_batch=tf.squeeze(input_batch,squeeze_dims=[3])
    if one_hot:
        input_batch=tf.one_hot(input_batch,depth=num_classes)

    return input_batch


def decode_labels(mask, num_images=1, num_classes=5):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


def save_img(out_path, img):
    #print img[0]
    img = np.reshape(img[0], [img.shape[1], img.shape[2]])
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, -1)
    img = decode_labels(img, 1, 5)
    re_img=np.reshape(img,[240, 240,3])
   #re_img = np.reshape(img, [320, 320, 3])
   # print re_img[100]

    #img = np.clip(re_img, 0, 1).astype(np.uint8)
    #print img
    scipy.misc.imsave(out_path, re_img)

if __name__ =="__main__":
    img=Image.open('/home/mi/PycharmProjects/Brain-tumor-segmention/data/BRATS2015_Training/IMGHGG/brats_tcia_pat153_0109/VSD.Brain_3more.XX.O.OT.42325/VSD.Brain_3more.XX.O.OT.42325.mha90.png')
    img= np.array(img.convert('L'))//63
    img=np.expand_dims(img,0)
    img=np.expand_dims(img,-1)
    img=decode_labels(img,1,5)

    save_img('../res/gt.png',img)


