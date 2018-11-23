"""Benchmark fully connected layer / matrix multiplication"""

import tensorflow as tf
import numpy as np


class dense_layer(object):
    """Class for gerenating the benchmark operations"""

    def __init__(self,
                 dim_input,
                 dim_output,
                 batchsize,
                 precision,
                 activation_fct,
                 optimizer,
                 devlist):
        """Initialize gemm

        Args:
            dim_input: Size of input vector / number of input features (int)
            dim_output: Size of output vector / number of output features (int)
            batchsize: (int)
            precision: bit depth (16/32/64)
            optimizer: (string)
            devlist: List of GPUs / CPUs (list)
        """

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.batchsize = batchsize
        self.precision = precision
        self.activation_fct = activation_fct
        self.opt = optimizer
        self.devlist = devlist


    def create_benchmark_op(self):
        """Create benchmark operation

        Returns:
            prog.op: Operation for multiplying two matrices
            g: TensorFlow graph
        """

        datatype = eval('tf.float%d' %(self.precision))
        act = eval(self.activation_fct)

        g = tf.Graph()
        with g.as_default():
            for dev in self.devlist:
                with tf.device(dev):
                    VecIn = tf.Variable(tf.ones(
                            dtype=datatype,
                            shape=[self.batchsize,self.dim_input]))
                    dense = tf.layers.dense(
                            inputs=VecIn,
                            units=self.dim_output,
                            kernel_initializer=tf.ones_initializer(),
                            activation = act)

        return dense.op, g


    def create_benchmark_op_with_backprop(self):
        """Create benchmark operation using tf.layer

        Returns:
            conv.op: Operation for convolution
            g: TensorFlow graph
        """

        datatype = eval('tf.float%d' %(self.precision))
        act = eval(self.activation_fct)
        opt = eval('tf.train.%s' % self.opt)

        g = tf.Graph()
        run_metadata = tf.RunMetadata()
        with g.as_default():
            for dev in self.devlist:
                with tf.device(dev):
                    VecIn = tf.Variable(tf.ones(
                            dtype=datatype,
                            shape=[self.batchsize,self.dim_input]))

                    target = tf.Variable(
                            tf.ones([
                                    self.batchsize,
                                    self.dim_output],
                            dtype=datatype))

                    dense = tf.layers.dense(
                            inputs=VecIn,
                            units=self.dim_output,
                            kernel_initializer=tf.ones_initializer(),
                            activation = act)

                    loss = tf.reduce_mean( tf.square( dense - target ) )
                    train_op = opt.minimize(loss=loss)

        return train_op, g
