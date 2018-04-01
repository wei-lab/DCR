#!/usr/bin/python
#!-*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def computeloss( infer_batch, label_batch ):
    label_batch = tf.one_hot( label_batch, depth=5 ,axis=-1 )
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=infer_batch, labels=label_batch);
    return tf.reduce_mean( loss )


def maskloss( infer_batch, label_batch ):
    loss = tf.square( 
                tf.subtract(
                    tf.cast( infer_batch, tf.float32 ),
                    tf.cast( label_batch, tf.float32 ) 
                ) 
            )
    return tf.reduce_mean( loss )


def optimizer( cost, learning_rate ):
    return tf.train.AdamOptimizer(learning_rate).minimize(cost)


def focal_loss( infer_batch, label_batch ):
    t = tf.squeeze(
            tf.one_hot(
                tf.cast(label_batch, tf.int32), depth=5, axis=-1 ), [3] )
    alpha = 0.75
    gamma = 2
    p = tf.nn.softmax( infer_batch )
    pt = tf.where( tf.equal(t, 0), 1 - p, p )
    w = alpha * t + (1 - alpha) * (1 - t)
    w = w * tf.pow( (1 - pt), gamma )
    loss = tf.nn.weighted_cross_entropy_with_logits( t, infer_batch, w )
    return tf.reduce_mean( loss )