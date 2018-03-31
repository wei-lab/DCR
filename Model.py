#!/usr/bin/python
#!-*- coding:utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class Model( object ):

    def _conv2d( self, x_tensor, num_outputs, conv_ksize, strides=(1, 1), padding='valid', rate=(1, 1),  name=None ):
        return tf.layers.conv2d( x_tensor, num_outputs, conv_ksize, 
                                 strides=strides, 
                                 padding=padding, 
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 dilation_rate=rate,
                                 name=name )


    def _deconv2d( self, x_tensor, num_outputs, conv_ksize, strides=(1, 1), padding='valid', name=None ):
        return tf.layers.conv2d_transpose( x_tensor, num_outputs, conv_ksize, 
            strides=strides, 
            padding=padding, 
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name=name)


    def _deconv2d_with_relu( self, x_tensor, num_outputs, conv_ksize, strides=(1, 1), padding='valid', name=None ):
        res = tf.layers.conv2d_transpose( x_tensor, num_outputs, conv_ksize, 
            strides=strides, 
            padding=padding, 
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name=name)
        return tf.nn.relu( res )


    def _max_pool( self, x_tensor, pool_ksize, strides=None, padding='valid', name=None ):
        if strides == None:
            strides = pool_ksize
        return tf.layers.max_pooling2d( x_tensor, pool_ksize, strides, padding=padding, name=name )


    def _avg_pool( self, x_tensor, pool_ksize, strides=None, padding='valid', name=None ):
        if strides == None:
            strides = pool_ksize
        return tf.layers.avg_pooling2d( net, pool_ksize, strides, padding=padding, name=name)


    def _bn( self, x_tensor, is_training=True, name=None ):
        return tf.layers.batch_normalization( x_tensor, training=is_training, name=name )


    def _relu( self, x_tensor, name=None ):
        return tf.nn.relu( x_tensor, name=name )


    def _conv_layer( self, x_tensor, num_outputs, ksize, strides=(1, 1), padding='valid', rate=(1, 1), name=None ):
        net = tf.layers.conv2d( x_tensor, num_outputs, ksize, 
                                 strides=strides, 
                                 padding=padding, 
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 dilation_rate=rate,
                                 name=name )
        return tf.nn.relu( net )


    def identity_block( self, x, kernel_size, filters, stage, block, is_training=True ):
        filters1, filters2, filters3 = filters
        conv_name_base = 'conv' + str(stage) + block + '_Res'
        bn_name_base = 'bn' + str(stage) + block + '_Res'

        net = self._conv2d( x, filters1, (1, 1), name='{}_2a'.format(conv_name_base) )
        net = self._bn( net, is_training=is_training, name='{}_2a'.format(bn_name_base) )
        net = self._relu( net )

        net = self._conv2d( net, filters2, kernel_size, padding='same', name='{}_2b'.format(conv_name_base) )
        net = self._bn( net, is_training=is_training, name='{}_2b'.format(bn_name_base) )
        net = self._relu( net )

        net = self._conv2d( net, filters3, (1, 1), name='{}_2c'.format(conv_name_base) )
        net = self._bn( net, is_training=is_training, name='{}_2c'.format(bn_name_base) )

        net = tf.add( net, x )
        net = self._relu( net )
        return net


    def conv_block( self, x, kernel_size, filters, stage, block, strides=(2, 2), is_training=True ):
        filters1, filters2, filters3 = filters
        conv_name_base = 'conv' + str(stage) + block + '_Res'
        bn_name_base = 'bn' + str(stage) + block + '_Res'

        net = self._conv2d( x, filters1, (1, 1), strides=strides, name='{}_2a'.format(conv_name_base) )
        net = self._bn( net, is_training=is_training, name='{}_2a'.format(bn_name_base) )
        net = self._relu( net )

        net = self._conv2d( net, filters2, kernel_size, padding='SAME', name='{}_2b'.format(conv_name_base) )
        net = self._bn( net, is_training=is_training, name='{}_2b'.format(bn_name_base) )
        net = self._relu( net )

        net = self._conv2d( net, filters3, (1, 1), name='{}_2c'.format(conv_name_base) )
        net = self._bn( net, is_training=is_training, name='{}_2c'.format(bn_name_base) )

        shortcut = self._conv2d( x, filters3, (1, 1), strides=strides, name='{}_1'.format(conv_name_base) )
        shortcut = self._bn( shortcut, is_training=is_training, name='{}_1'.format(bn_name_base) )

        net = tf.add( net, shortcut )
        net = self._relu( net )
        return net


    def ResNet50( self, x, base=128, is_training=True ):
        net = self._conv2d( x, 64, (7, 7), padding='SAME', name='conv1' )
        net = self._bn( net, is_training=is_training, name='bn_conv1' )
        net = self._relu( net )
        net = self._max_pool( net, (3, 3), strides=(2, 2) )

        net = self.conv_block( net, 3, [64, 64, base], stage=2, block='a' )
        net = self.identity_block( net, 3, [64, 64, base], stage=2, block='b' )
        sub1 = self.identity_block( net, 3, [64, 64, base], stage=2, block='c' )

        net = self.conv_block( sub1, 3, [128, 128, base * 2], stage=3, block='a' )
        net = self.identity_block( net, 3, [128, 128, base * 2], stage=3, block='b' )
        net = self.identity_block( net, 3, [128, 128, base * 2], stage=3, block='c' )
        sub2 = self.identity_block( net, 3, [128, 128, base * 2], stage=3, block='d' )

        net = self.conv_block( sub2, 3, [256, 256, base * 4], stage=4, block='a' )
        net = self.identity_block( net, 3, [256, 256, base * 4], stage=4, block='b' )
        net = self.identity_block( net, 3, [256, 256, base * 4], stage=4, block='c' )
        net = self.identity_block( net, 3, [256, 256, base * 4], stage=4, block='d' )
        net = self.identity_block( net, 3, [256, 256, base * 4], stage=4, block='e' )
        sub3 = self.identity_block( net, 3, [256, 256, base * 4], stage=4, block='f' )

        net = self.conv_block( sub3, 3, [512, 512, base * 8], stage=5, block='a') 
        net = self.identity_block( net, 3, [512, 512, base * 8], stage=5, block='b' )
        sub4 = self.identity_block( net, 3, [512, 512, base * 8], stage=5, block='c' )

        return sub1, sub2, sub3, sub4


    def _dcr_block_a( self, x, num_filter, para, stage, block, is_training=True ):
        # para : a 3-dim tuple(or list) like structure 
        #   para[0] - dilation rate
        #   para[1] - the kernel size of the left term which is a 1-by-1 convolution
        #   para[2] - the kernel size of the right term which is dilated convolution 
        rate = para[0]
        l_kconv = para[1]
        r_kconv = para[2]

        conv_name_base = 'conv' + str(stage) + block + '_DCR_A'
        bn_name_base = 'bn' + str(stage) + block + '_DCR_A'

        net_l = self._conv2d( x, num_filter, l_kconv, padding='SAME', name='{}_l'.format(conv_name_base) )
        net_l = self._bn( net_l, is_training=is_training, name='{}_l'.format(bn_name_base) )

        net_r = self._conv2d( x, num_filter, r_kconv, padding='SAME', rate=rate, name='{}_r'.format(conv_name_base) )
        net_r = self._bn( net_r, is_training=is_training, name='{}_r'.format(bn_name_base) )

        net = self._relu( tf.add( net_l, net_r ) )
        return net


    def _dcr_block_b( self, x, num_filter, para, stage, block, is_training=True ):
        kconv = para[1]

        conv_name_base = 'conv' + str(stage) + block + '_DCR_B'
        bn_name_base = 'bn' + str(stage) + block + '_DCR_B'

        net = self._conv2d( x, num_filter, kconv, padding='SAME', name='{}_term'.format(conv_name_base) )
        net = self._bn( net, is_training=is_training, name='{}_term'.format(bn_name_base) )
        net = self._relu( net )
        return net 


    def _dcr_block_c( self, x, num_filter, para, stage, block, is_training=True ):
        rate = para[0]
        kconv = para[2]

        conv_name_base = 'conv' + str(stage) + block + '_DCR_C'
        bn_name_base = 'bn' + str(stage) + block + '_DCR_C'

        net = self._conv2d( x, num_filter, kconv, padding='SAME', rate=rate, name='{}_term'.format(conv_name_base) )
        net = self._bn( net, is_training=is_training, name='{}_term'.format(bn_name_base) )
        net = self._relu( net )
        return net 


    def DCR( self, inputs_x, filters, dcr_type, dilated_rates=None ):
        '''
            DCR
            type = A, B, C
        '''
        dcr_type = dcr_type.upper()

        assert dcr_type in ['A', 'B', 'C']
        if dcr_type in ['A', 'C']:
            assert isinstance(dilated_rates, list) or isinstance(dilated_rates, tuple) and len(dilated_rates) == 4
            r1, r2, r3, r4 = dilated_rates
        else:
            r1 = r2 = r3 = r4 = 0

        dcr_list = { 'A' : self._dcr_block_a,
                     'B' : self._dcr_block_b,
                     'C' : self._dcr_block_c }
        _dcr_block = dcr_list.get( dcr_type )

        s1, s2, s3, s4 = inputs_x
        f1, f2, f3, f4 = filters

        dcr1 = _dcr_block( s1, f1, [r1, 1, 3], stage=2, block='a' )
        dcr1 = self._conv2d( dcr1, f1, 1, name='conv2b_after_DCR_A' )

        dcr2 = _dcr_block( s2, f2, [r2, 1, 3], stage=3, block='a' )
        dcr2 = self._conv2d( dcr2, f2, 1, name='conv3b_after_DCR_A' )

        dcr3 = _dcr_block( s3, f3, [r3, 1, 3], stage=4, block='a' )
        dcr3 = self._conv2d( dcr3, f3, 1, name='conv4b_after_DCR_A' )

        dcr4 = _dcr_block( s4, f4, [r4, 1, 3], stage=5, block='a' )
        dcr4 = self._conv2d( dcr4, f4, 1, name='conv5b_after_DCR_A' )

        return dcr1, dcr2, dcr3, dcr4


    def Transpose( self, inputs_dcr ):
        d1, d2, d3, d4 = inputs_dcr
        base_name = 'deconv{}'
        net = self._deconv2d( d4, 64, 3, strides=2, padding='SAME', name=base_name.format(5) )
        print(net)

        net = tf.add( net, d3 )
        net = self._deconv2d( net, 64, 3, strides=2, padding='SAME', name=base_name.format(4) )
        print(net)

        net = tf.add( net, d2 )
        net = self._deconv2d( net, 64, 3, strides=2, padding='SAME', name=base_name.format(3) )
        print(net)

        net = tf.add( net, d1 )
        net = self._deconv2d( net, 64, 3, strides=2, padding='SAME', name=base_name.format(2) )
        print(net)

        net = self._deconv2d( net, 64, 3, strides=2, padding='SAME', name=base_name.format(1) )
        print(net)

        net = self._conv2d( net, 5, 1, name='restore_image' )
        print(net)
        return net


    def ResNet_DCR_Model( self, x, dcr_type, dilated_rates=None ):
        '''
            dilated_rates:
                [12, 6, 4, 1]
                [8, 4, 2, 1]
                [6, 3, 2, 1]
            self.DCR:
                'A', 'B', 'C'
        '''
        s1, s2, s3, s4 = self.ResNet50( x )
        d1, d2, d3, d4 = self.DCR( [s1, s2, s3, s4], [64 for i in range(4)], dcr_type, dilated_rates=dilated_rates )
        res = self.Transpose( [d1, d2, d3, d4] )
        return res


    def FCN( self, x ):
        # x  -  224 * 224

        # block 1
        net = self._conv_layer( x, 64, 3, padding='SAME', name='conv1_1' )
        net = self._conv_layer( net, 64, 3, padding='SAME', name='conv1_2' )
        net = self._max_pool( net, 2, padding='SAME', name='pool1' )
        # 112 * 112

        #block 2
        net = self._conv_layer( net, 128, 3, padding='SAME', name='conv2_1' )
        net = self._conv_layer( net, 128, 3, padding='SAME', name='conv2_2' )
        net = self._max_pool( net, 2, padding='SAME', name='pool2' )
        # 56 * 56

        # block 3
        net = self._conv_layer( net, 256, 3, padding='SAME', name='conv3_1' )
        net = self._conv_layer( net, 256, 3, padding='SAME', name='conv3_2' )
        net = self._conv_layer( net, 256, 3, padding='SAME', name='conv3_3' )
        score_pool3 = self._max_pool( net, 2, padding='SAME', name='pool3' )
        # print( score_pool3 )
        # 28 * 28

        # block 4
        net = self._conv_layer( score_pool3, 256, 3, padding='SAME', name='conv4_1' )
        net = self._conv_layer( net, 256, 3, padding='SAME', name='conv4_2' )
        net = self._conv_layer( net, 256, 3, padding='SAME', name='conv4_3' )
        score_pool4 = self._max_pool( net, 2, padding='SAME', name='pool4' )
        # print( score_pool4 )
        # 14 * 14

        # block 5
        net = self._conv_layer( score_pool4, 256, 3, padding='SAME', name='conv5_1' )
        net = self._conv_layer( net, 256, 3, padding='SAME', name='conv5_2' )
        net = self._conv_layer( net, 256, 3, padding='SAME', name='conv5_3' )
        score_pool5 = self._max_pool( net, 2, padding='SAME', name='pool5' )
        # print( score_pool5 )
        # 7 * 7

        upsampled5 = self._deconv2d( score_pool5, 256, 3, strides=2, padding='SAME', name='upsampled5' )
        add_4and5 = tf.add( score_pool4, upsampled5 )
        # 14 * 14

        upsampled4 = self._deconv2d( add_4and5, 256, 3, strides=2, padding='SAME', name='upsampled4' )
        add_3and4 = tf.add( score_pool3, upsampled4 )
        # 28 * 28

        # for add_3and4, to upsampled 8x
        upsampled_8x = self._deconv2d_with_relu( add_3and4, 128, 3, strides=2, padding='SAME', name='upsampled3_8x_a' ) # 56 * 56
        upsampled_8x = self._deconv2d_with_relu( upsampled_8x, 64, 3, strides=2, padding='SAME', name='upsampled3_8x_b' ) # 112 * 112
        upsampled_8x = self._deconv2d( upsampled_8x, 64, 3, strides=2, padding='SAME', name='upsampled3_8x_c' ) # 224 * 224

        # for add_4and5, to upsampled 16x
        upsampled_16x = self._deconv2d_with_relu( add_4and5, 128, 3, strides=2, padding='SAME', name='upsampled3_16x_a' )# 28 * 28
        upsampled_16x = self._deconv2d_with_relu( upsampled_16x, 64, 3, strides=2, padding='SAME', name='upsampled3_16x_b' ) # 56 * 56
        upsampled_16x = self._deconv2d_with_relu( upsampled_16x, 64, 3, strides=2, padding='SAME', name='upsampled3_16x_c' ) # 112 * 112
        upsampled_16x = self._deconv2d( upsampled_16x, 64, 3, strides=2, padding='SAME', name='upsampled3_16x_d' ) # 224 * 224

        net = tf.add( upsampled_8x, upsampled_16x)

        # for score_pool5, to upsampled 32x
        upsampled_32x = self._deconv2d_with_relu( score_pool5, 128, 3, strides=2, padding='SAME', name='upsampled3_32x_a' ) # 14 * 14
        upsampled_32x = self._deconv2d_with_relu( upsampled_32x, 64, 3, strides=2, padding='SAME', name='upsampled3_32x_b' ) # 28 * 28
        upsampled_32x = self._deconv2d_with_relu( upsampled_32x, 64, 3, strides=2, padding='SAME', name='upsampled3_32x_c' ) # 56 * 56
        upsampled_32x = self._deconv2d_with_relu( upsampled_32x, 64, 3, strides=2, padding='SAME', name='upsampled3_32x_d' ) # 112 * 112
        upsampled_32x = self._deconv2d( upsampled_32x, 64, 3, strides=2, padding='SAME', name='upsampled3_32x_e' ) # 224 * 224

        net = tf.add( net, upsampled_32x )
        net = self._conv2d( net, 5, 1, name='fcn_output')
        return net


    def Unet( self, x, keep_prob=0.5 ):
        # 224 * 224

        # stage 1
        conv1 = self._conv_layer( x, 64, 3, padding='SAME', name='stage_1_conv_a' )
        conv1 = self._conv_layer( conv1, 64, 3, padding='SAME', name='stage_1_conv_b' ) # 224
        pool1 = self._max_pool( conv1, 2, padding='SAME', name='stage_1_pool' ) # 112

        # stage 2
        conv2 = self._conv_layer( pool1, 128, 3, padding='SAME', name='stage_2_conv_a' )
        conv2 = self._conv_layer( conv2, 128, 3, padding='SAME', name='stage_2_conv_b' ) # 112
        pool2 = self._max_pool( conv2, 2, padding='SAME', name='stage_2_pool' ) # 56

        # stage 3
        conv3 = self._conv_layer( pool2, 256, 3, padding='SAME', name='stage_3_conv_a' )
        conv3 = self._conv_layer( conv3, 256, 3, padding='SAME', name='stage_3_conv_b' ) # 56
        pool3 = self._max_pool( conv3, 2, padding='SAME', name='stage_3_pool' ) # 28

        # stage 4
        conv4 = self._conv_layer( pool3, 512, 3, padding='SAME', name='stage_4_conv_a' )
        conv4 = self._conv_layer( conv4, 512, 3, padding='SAME', name='stage_4_conv_b' )
        # drop4 = tf.nn.dropout( conv4, keep_prob=keep_prob )
        pool4 = self._max_pool( drop4, 2, padding='SAME', name='stage_4_pool' ) # 14

        # stage 5
        conv5 = self._conv_layer( pool4, 1024, 3, padding='SAME', name='stage_5_conv_a' )
        conv5 = self._conv_layer( conv5, 1024, 3, padding='SAME', name='stage_5_conv_b' ) # 14
        # drop5 = tf.nn.dropout( conv5, keep_prob=keep_prob )
        deconv5 = self._deconv2d_with_relu( conv5, 512, 3, strides=2, padding='SAME', name='stage_5_deconv' ) # 28

        # stage 6
        merge6 = tf.concat( [drop4, deconv5], -1, name='merge6' )
        conv6 = self._conv_layer( merge6, 512, 3, padding='SAME', name='stage_6_conv_a' )
        conv6 = self._conv_layer( merge6, 512, 3, padding='SAME', name='stage_6_conv_b' )
        deconv6 = self._deconv2d_with_relu( conv6, 256, 3, strides=2, padding='SAME', name='stage_6_deconv' ) # 56 

        # stage 7
        merge7 = tf.concat( [conv3, deconv6], -1, name='merge7' )
        conv7 = self._conv_layer( merge7, 256, 3, padding='SAME', name='stage_7_conv_a' )
        conv7 = self._conv_layer( conv7, 256, 3, padding='SAME', name='stage_7_conv_b' )
        deconv7 = self._deconv2d_with_relu( conv7, 128, 3, strides=2, padding='SAME', name='stage_7_deconv' ) # 112

        # stage 8
        merge8 = tf.concat( [conv2, deconv7], -1, name='merge8' )
        conv8 = self._conv_layer( merge8, 128, 3, padding='SAME', name='stage_8_conv_a' )
        conv8 = self._conv_layer( conv8, 128, 3, padding='SAME', name='stage_8_conv_b' )
        deconv8 = self._deconv2d_with_relu( conv8, 64, 3, strides=2, padding='SAME', name='stage_8_deconv' ) # 224

        # stage 9
        merge9 = tf.concat( [conv1, deconv8], -1, name='merge9' )
        conv9 = self._conv_layer( merge9, 64, 3, padding='SAME', name='stage_9_conv_a' )
        conv9 = self._conv_layer( conv9, 64, 3, padding='SAME', name='stage_9_conv_b' )
        conv9 = self._conv_layer( conv9, 5, 3, padding='SAME', name='output' )

        return conv9



if __name__ == '__main__':
    x = tf.placeholder( tf.float32, [None, 224, 224, 4] )

    # k = Model().ResNet_DCR_Model( x, 'A', [3, 3, 3, 3])
    # print(k)

    # fcn = Model().FCN( x )
    # print( fcn )

    unet = Model().Unet( x )
    print( unet )