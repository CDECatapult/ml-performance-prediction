"""Benchmark convolution"""

import tensorflow as tf
import numpy as np


class convolution(object):
    """Class for gerenating the benchmark operations"""

    def __init__(self,
                 batchsize,
                 matsize,
                 kernelsize,
                 channels_in,
                 channels_out,
                 strides,
                 precision,
                 padding,
                 activation_fct,
                 use_bias,
                 devlist,
                 optimizer):
        """Initialize convolution

        Args:
            args: Input arguments
            devlist: List of GPUs / CPUs (list)
        """

        self.matsize = matsize
        self.kernelsize = kernelsize
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.strides = strides
        self.batchsize = batchsize
        self.padding = padding
        self.use_bias = use_bias
        self.precision = precision
        self.devlist = devlist
        self.activation_fct = activation_fct
        self.opt = optimizer


    def create_benchmark_op(self):
        """Create benchmark operation using tf.layer

        Returns:
            conv.op: Operation for convolution
            g: TensorFlow graph
        """

        datatype = eval('tf.float%d' %(self.precision))
        act = eval(self.activation_fct)

        g = tf.Graph()
        run_metadata = tf.RunMetadata()
        with g.as_default():
            for dev in self.devlist:
                with tf.device(dev):
                    matA = tf.Variable(
                            tf.ones([
                                    self.batchsize,
                                    self.matsize,
                                    self.matsize,
                                    self.channels_in],
                            dtype=datatype))
                    conv = tf.layers.conv2d(
                            inputs=matA,
                            filters=self.channels_out,
                            kernel_size=[self.kernelsize,self.kernelsize],
                            strides=self.strides,
                            padding=self.padding,
                            activation = act,
                            use_bias=self.use_bias)

        return conv.op, g


    def create_benchmark_op_with_backprop(self):
        """Create benchmark operation using tf.layer

        Returns:
            conv.op: Operation for convolution
            g: TensorFlow graph
        """

        datatype = eval('tf.float%d' %(self.precision))
        opt = eval('tf.train.%s' % self.opt)
        act = eval(self.activation_fct)

        if self.padding == 'SAME':
            target_size = np.ceil(np.float(self.matsize)/self.strides)
        else:
            target_size = np.ceil(np.float((self.matsize-(self.kernelsize-1)))/self.strides)

        g = tf.Graph()
        run_metadata = tf.RunMetadata()
        with g.as_default():
            for dev in self.devlist:
                with tf.device(dev):
                    matA = tf.Variable(
                            tf.ones([
                                    self.batchsize,
                                    self.matsize,
                                    self.matsize,
                                    self.channels_in],
                            dtype=datatype))

                    target = tf.Variable(
                            tf.ones([
                                    self.batchsize,
                                    target_size,
                                    target_size,
                                    self.channels_out],
                            dtype=datatype))

                    conv = tf.layers.conv2d(
                            inputs=matA,
                            filters=self.channels_out,
                            kernel_size=[self.kernelsize,self.kernelsize],
                            strides=self.strides,
                            padding=self.padding,
                            activation = act,
                            use_bias=self.use_bias)

                    loss = tf.reduce_mean( tf.square( conv - target ) )
                    train_op = opt.minimize(loss=loss)

        return train_op, g
