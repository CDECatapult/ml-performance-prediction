"""Benchmark convolution"""

import tensorflow as tf


class convolution(object):
    """Class for gerenating the benchmark operations"""

    def __init__(self, args, devlist):
        """Initialize convolution

        Args:
            args: Input arguments
            devlist: List of GPUs / CPUs (list)
        """

        self.matsize = args.matsize
        self.kernelsize = args.kernelsize
        self.channels_in = args.channels_in
        self.channels_out = args.channels_out
        self.batchsize = args.batchsize
        self.padding = args.padding
        self.precision = args.precision
        self.comment = args.comment
        self.devlist = devlist


    def create_benchmark_op1(self):
        """Create benchmark operation using tf.layer

        Returns:
            conv.op: Operation for convolution
            g: TensorFlow graph
        """

        datatype = eval('tf.float%d' %(self.precision))

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
                            kernel_size=self.kernelsize,
                            padding=self.padding)

        return conv.op, g

    def create_benchmark_op2(self):
        """Create benchmark operation using tf.nn

        Returns:
            conv.op: Operation for convolution
            g: TensorFlow graph
        """

        datatype = eval('tf.float%d' %(self.precision))

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
                    kernel = tf.Variable(
                            tf.ones([
                                    self.kernelsize,
                                    self.kernelsize,
                                    self.channels_in,
                                    self.channels_out],
                            dtype=datatype))
                    conv = tf.nn.conv2d(
                            input=matA,
                            filter=kernel,
                            strides=[1,1,1,1],
                            padding=self.padding)

        return conv.op, g


    def generate_logtext(self, timeUsed, ops, mem):
        """Function that generates comma separated text for a logfile"""

        logtext = ('convolution, %d, %d, %d, %d, %d, %d, %.3f, %.3f, %.3f, %s\n'
                %(self.matsize,
                self.batchsize,
                self.kernelsize,
                self.channels_in,
                self.channels_out,
                self.precision,
                timeUsed*1000,
                ops*1e-9/timeUsed,
                mem/1e6,
                self.comment))

        return logtext
