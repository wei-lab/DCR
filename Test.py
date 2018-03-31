#!/usr/bin/python
#!-*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from TF_Input import net_inputs_test
from Model import Model
from sys import stdout
import SimpleITK as sitk
import tensorflow as tf
import numpy as np
import scipy
import random
import os
import utils


test_data_dir = '../data/tfrecord/test.tfrecords'
dcr_type = 'A'
dilated_rates = [12, 6, 4, 1]
CKPT_PATH = 'ckpt_dcr_A'
LOG_DIR = 'logs_dcr_A'
batch_size = 1
test_file = 'testing.txt'

IMG_DIR = '../images/img_dcr_{}_test'.format( dcr_type )
MHA_DIR = '../mha/mha_dcr_{}_test'.format( dcr_type )

IMG_DIR = IMG_DIR.strip().rstrip('/')
MHA_DIR = MHA_DIR.strip().rstrip('/')

if not os.path.exists( MHA_DIR ):
    os.makedirs( MHA_DIR )
    print( 'INFO : MHA_DIR does not exists, create a directory!' )


def from_np_to_mha( images, mha_name ):
    img = sitk.GetImageFromArray( images )
    img.SetOrigin( [0, 0, 0] )
    img.SetSpacing( [1.0, 1.0, 1.0] )
    sitk.WriteImage( img, mha_name )


def read_mha_files( mha_file_name ):
    file_list = []
    for line in open(mha_file_name).readlines():
        file_list.append( line )
    return file_list[::-1]


def random_rotation( input_img, label_img ):
    angle = random.randint( -45, 45 )
    output_img = tf.contrib.image.rotate( input_img, angle )
    output_label = tf.contrib.image.rotate( label_img, angle )
    return output_img, output_label


def main( test_data_dir, save_img=False ):
    if save_img and not os.path.exists( IMG_DIR ):
        os.makedirs( IMG_DIR )
        print( 'INFO : IMG_DIR does not exists, create a directory!' )

    mha_list = read_mha_files( test_file )
    length = len( mha_list ) * 155

    image_batch = net_inputs_test( batch_size, test_data_dir )
    image_batch = tf.cast( image_batch, tf.float32 )

    output = Model( image_batch, dcr_type, dilated_rates=dilated_rates )
    out_mask = tf.expand_dims( tf.cast( tf.arg_max(output, dimension=3), tf.uint8 ), -1 )
    out_mask = tf.image.resize_images( out_mask, [240, 240] )


    with tf.Session() as sess:
        mha = []

        ckpt = tf.train.get_checkpoint_state( CKPT_PATH )
        if not ckpt:
            raise RuntimeError('No Checkpoint Found !')
        else:
            saver=tf.train.Saver()
            ckpt_path = ckpt.model_checkpoint_path
            saver.restore( sess, ckpt_path )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess=sess, coord=coord )

        for step in range( 1, length + 1 ):
            stdout.write( '\rProcessing {} / {} ...'.format(step, length) )

            _out_mask = sess.run( out_mask )
            _out_mask = np.round(_out_mask).astype(np.uint8)
            re_img = _out_mask.reshape( [240, 240, 1] )
            mha.append( re_img )

            if save_img:
                img = utils.decode_labels( np.round(_out_mask).astype(np.uint8) ).reshape([240, 240, 3])
                scipy.misc.imsave( '{}/{}.jpg'.format(IMG_DIR, step), img )

            if step % 155 == 0 and step != 0:
                file_id = mha_list.pop().split(',')[0].strip()[1:-1].split('.')[-2]
                mha_name = ''.join( ['VSD.Seg_HG_001.', str(file_id), '.mha'] )
                mha = []

        coord.request_stop()
        coord.join( threads )
        print() 


if __name__ == '__main__':
    main( test_data_dir, save_img=False )
