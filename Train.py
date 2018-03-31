#!/usr/bin/python
#!-*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Model import Model
from Optimization import computeloss, optimizer, maskloss
from TF_Input import net_inputs
from sys import stdout
from time import time
import tensorflow as tf
import os


'''
dcr struct type:
    'A'  for DCR_A
    'B'  for DCR_B
    'C'  for DCR_C

dilated rates: 
    [12, 6, 4, 1] for DCR_A
    [8, 4, 2, 1]  for DCR_D
    [6, 3, 2, 1]  for DCR_E

    if use DCR_C, the dilated is same as DCR_A
    if use DCR_B, the dilated_rates = None
'''

dcr_type = 'A'
dilated_rates = [12, 6, 4, 1]
CKPT_PATH = 'ckpt_dcr_A'
LOG_DIR = 'logs_dcr_A'
GPU_MEM_RATE = 0.3

CKPT_PATH = CKPT_PATH.strip().rstrip('\\')
LOG_DIR = LOG_DIR.strip().rstrip('\\')

if not os.path.exists( CKPT_PATH ):
    os.makedirs( CKPT_PATH )
    print( 'INFO : CKPT_PATH does not exists, create a directory!' )
if not os.path.exists( LOG_DIR ):
    os.makedirs( LOG_DIR )
    print( 'INFO : LOG_DIR does not exists, create a directory!' )


epochs = 100
batch_size = 1
learning_rate = 1e-4
batchs = 155 * 219
if batchs % batch_size:
    batchs = batchs // batch_size
else:
    batchs = batchs // batch_size + 1


def train():
    image_batch, label_batch = net_inputs( batch_size, '../data/tfrecord/train.tfrecords' )

    image_batch = tf.cast( image_batch, tf.float32 )
    label_batch = tf.cast( label_batch, tf.int32 )

    logits = Model().ResNet_DCR_Model( image_batch, dcr_type, dilated_rates=dilated_rates )
    out_mask = tf.cast( tf.arg_max(logits, dimension=3), tf.int32 )
    cost = computeloss( logits, label_batch ) + maskloss( out_mask, label_batch )

    tf.summary.scalar( 'cost', cost )
    optimizator = optimizer( cost, learning_rate )
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_RATE

    with tf.Session(config=config) as sess:

        ckpt = tf.train.get_checkpoint_state( CKPT_PATH )

        if not ckpt:
            print( 'No checkpoint file found. Initializing...' )
            global_step = 0
            sess.run( tf.global_variables_initializer() )
        else:
            global_step = len( ckpt.all_model_checkpoint_paths )
            _ckpt_path = ckpt.model_checkpoint_path
            saver.restore( sess, _ckpt_path )

        train_writer = tf.summary.FileWriter( LOG_DIR, sess.graph )
        merged = tf.summary.merge_all()

        total_start = time()
        for epoch in range( epochs ):
            epoch_start = time()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners( sess=sess, coord=coord )
            for batch in range( batchs ):
                batch_start = time()
                train_cost, out_mask_value, _ = sess.run( [cost, out_mask, optimizator] )
                batch_time_use = time() - batch_start

                stdout.write( '\rEpoch {:>4}, Batch {:>5}/{:>5} - Loss: {:>.8f}. Time use: {:>.4}s'.format(epoch + 1, batch + 1, batchs, train_cost, batch_time_use ) )

                if global_step % 300:
                    summary_str = sess.run( merged )
                    train_writer.add_summary( summary_str, global_step )

                if global_step % 1000 == 0:
                    epoch_time_use = time() - epoch_start
                    print( "\n---- Global Step: {} ---- Epoch: {:>4}, Batch {:>5} ----".format(global_step, epoch + 1, batch + 1) )
                    print( "    Train Cost: {:>.8f}, Epoch time use: {:>.4}s".format(train_cost, epoch_time_use) )
                    print()
                    saver.save( sess, os.path.join(CKPT_PATH, 'model.ckpt'), global_step=global_step )

                global_step += 1

            saver.save(sess, os.path.join(CKPT_PATH, 'model.ckpt'), global_step=global_step)

            total_time_use = time() - total_start
            print( "\n---- Global Step: {}, Finish Training! ----".format(global_step) )
            print( "    Train Cost: {:>.8f}, Epoch time use: {:>.4}s".format(train_cost, total_time_use) )
            print()
            # Train finish
        coord.request_stop()
        coord.join( threads )


if __name__ == '__main__':
    train()