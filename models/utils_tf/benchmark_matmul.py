"""Benchmark matrix multiplication"""

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import time
from utils_tf import merge_timeline

class gemm(object):
    """Class for gerenating the benchmark operations"""

    def __init__(self, args, devlist):
        """Initialize gemm

        Args:
            args: Input arguments
            devlist: List of GPUs / CPUs (list)
        """

        self.matsize = args.matsize
        self.precision = args.precision
        self.comment = args.comment
        self.devlist = devlist


    def create_benchmark_op(self):
        """Create benchmark operation

        Returns:
            prog.op: Operation for multiplying two matrices
            g: TensorFlow graph
        """

        datatype = eval('tf.float%d' %(self.precision))

        g = tf.Graph()
        with g.as_default():
            for dev in self.devlist:
                with tf.device(dev):
                    matA = tf.Variable(
                            tf.ones([self.matsize,self.matsize],
                            dtype=datatype))
                    matB = tf.Variable(
                            tf.ones([self.matsize,self.matsize],
                            dtype=datatype))
                    prod = tf.matmul(matA,matB)

        return prod.op, g


    def generate_logtext(self, timeUsed, ops, mem):
        """Function that generates comma separated text for a logfile"""

        logtext = ('matrix multiplication, %d, %d, %.3f, %.3f, %s\n'
                %(self.matsize,
                self.precision,
                ops*1e-9/timeUsed,
                mem/1e6,
                self.comment))
        return logtext
