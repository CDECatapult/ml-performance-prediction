"""Benchmark script for frequently used machine learning
operations and VGG16 as example of a typical CNN.

Using TensorFlow and Keras with TensorFlow backend
"""
import os
import argparse
import tensorflow as tf
import time
from utils_tf import benchmark_matmul, benchmark_conv, benchmark_conv_mult, benchmark_VGG16
from utils_tf import run_benchmark


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


parser = argparse.ArgumentParser('Benchmarking different aspects of a machine learning algorithm')

# Benchmarks to perform
parser.add_argument('--testMatMul', action="store_true", default=False, help='Benchmark matrix multiplication')
parser.add_argument('--testConv', action="store_true", default=False, help='Benchmark 2D convolution')
parser.add_argument('--testVGG16', action="store_true", default=False, help='Benchmark training VGG-16 on sythetic data')

# General parameters
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--devlist', type=str, default='', help='List of devices to use, overwrites num_gpu if set')
parser.add_argument('--precision', type=int, default=32, help='Precision')
parser.add_argument('--logfile', type=str, default='', help='Text file to store results')
parser.add_argument('--device', type=str, default='', help='Device name as appearing in logfile')
parser.add_argument('--use_tf_profiler', action="store_true", default=False, help='Calculate number of FLOPs with ft profiler')
parser.add_argument('--iter_benchmark', type=int, default=200, help='Number of iterations for benchmark')
parser.add_argument('--iter_timeline', type=int, default=10, help='Number of iterations for timeline')
parser.add_argument('--iter_warmup', type=int, default=10, help='Number of iterations for warm-up')
parser.add_argument('--iter_internal', type=int, default=10, help='Number of iterations without transfering data')

parser.add_argument('--batchsize', type=int, default=128, help='Batch size for convolutions / training CNN')
parser.add_argument('--no_saving', action="store_true", default=False, help='save benchmarking results to csv')
parser.add_argument('--no_timeline', action="store_true", default=False, help='Make tf timeline')
parser.add_argument('--comment', type=str, default='', help='Comment in logfile')

# Parameters for matrix multiplication / 2D convolution
parser.add_argument('--logFLOPs', type=int, default=0, help='log10 of number of FLOPs to perform (will be rounded up), overwrites iter_benchmark, set 0 to use iter')
parser.add_argument('--matsize', type=int, default=1024, help='Size of each matrix for benchmark')
parser.add_argument('--kernelsize', type=int, default=3, help='Size of kernel for benchmarking convolution')
parser.add_argument('--channels_in', type=int, default=1, help='Number of layers going into a convolution')
parser.add_argument('--channels_out', type=int, default=1, help='Number of features of a convolution')
parser.add_argument('--padding', type=str, default='SAME', help='Padding for convolutional layers (SAME or VALID)')
parser.add_argument('--use_tf_layers', action="store_true", default=False, help='Use tf.layer for convolution')

# Parameters for CNNs
parser.add_argument('--imgsize', type=int, default=224, help='Size of (square) images')
parser.add_argument('--numclasses', type=int, default=1000, help='Number of image classes')
parser.add_argument('--optimizer', type=str, default='sgd', help='Optimzer used for VGG-16 (sgd or rmsprop)')

args = parser.parse_args()


def generate_devlist(devlist, num_gpu):
    """Creates list with devices

    Args:
        devlist: Comma separated list of devices, overwrites num_gpu
        num_gpu: Number of GPUs to be used

    Return:
        devlist: List of devices
        use_gpu: Whether GPUs are used (boolean)
    """
    if devlist=='':
        if num_gpu==0:
            devlist = ['/cpu:0']
            use_gpu = False
        else:
            devlist = ['/gpu:%d' %i for i in range(num_gpu)]
            use_gpu = True
    else:
        use_gpu = ('gpu' in devlist.lower())
        devlist = devlist.split(',')
    return devlist, use_gpu


def main(_):
    """Main function that runs all benchmarks"""

    devlist, use_gpu = generate_devlist(args.devlist, args.num_gpu)


    ########## Benchmark matrix-matrix multiplication ##########
    if args.testMatMul:
        if args.logfile == '':
            logfile = str('/results/benchmark_matmul_%s_%s'
                    %(args.device, time.strftime("%Y%m%d")))
        else:
            logfile = args.logfile

        ops = (args.matsize**3
                + (args.matsize)*args.matsize**2)
                # matsize**3 multiplications,
                # (matsize-1)*matsize**2 additions

        gemm = benchmark_matmul.gemm(args, devlist)

        gemm_op, gemm_graph = gemm.create_benchmark_op()

        bm_matmul = run_benchmark.benchmark(
                gemm_op,
                args.iter_warmup,
                args.iter_benchmark,
                args.iter_timeline,
                gemm_graph)
        print("========================================\n")
        print("Start matrix multiplication")
        timeUsed = bm_matmul.run_benchmark()
        print("\n%d x %d matrix multiplication (float%d): "
                "%.3f ms, %.3f GFLOPS (%.2f matrices per sec)"
                % (args.matsize,
                args.matsize,
                args.precision,
                timeUsed*1000,
                ops*1e-9/timeUsed,
                1/timeUsed))

        if not args.no_saving:
            if not os.path.isfile('%s.csv'%logfile):
                header = ('operation, matsize, precision (bits), '
                        'performance (GFLOPs/sec), memory use (MB), '
                        'comment \n')
                f = open('%s.csv'%logfile,'a+')
                f.write(header)
                f.close()

            if use_gpu:
                mem = bm_matmul.get_memory_use()
            else:
                mem = 0
            with open('%s.csv'%logfile,'a+') as f:
                f.write(gemm.generate_logtext(timeUsed, ops, mem))

        if not args.no_timeline:
            bm_matmul.run_timeline(logfile, args.batchsize)
        print("\n========================================\n\n")


    ########### Benchmark convolution ##########
    if args.testConv:
        if args.logfile == '':
            logfile = str('/results/benchmark_convolution_%s_%s'
                    %(args.device, time.strftime("%Y%m%d")))
        else:
            logfile = args.logfile

        ops = (args.batchsize
                * args.matsize**2
                * (2*args.kernelsize**2
                    * args.channels_in
                    * args.channels_out)
                )

        conv = benchmark_conv_mult.convolution(args, devlist)

        if args.use_tf_layers:
            conv_op, conv_graph = conv.create_benchmark_op1()
        else:
            conv_op, conv_graph = conv.create_benchmark_op2()

        bm_conv = run_benchmark.benchmark(
                conv_op,
                args.iter_warmup,
                args.iter_benchmark,
                args.iter_timeline,
                conv_graph)

        print("========================================\n")
        print("Start convolution")
        timeUsed =  bm_conv.run_benchmark()

        print("\n%d x %d x %d x %d convolution (float%d): "
                "%.3f ms, %.3f GFLOPS (%.2f matrices per sec)"
                % (args.matsize,
                args.kernelsize,
                args.channels_in,
                args.channels_out,
                args.precision,
                timeUsed*1000,
                ops*1e-9/timeUsed,
                1/timeUsed))

        if not args.no_saving:
            if not os.path.isfile('%s.csv'%logfile):
                header = ('operation, matsize, batchsize, kernelsize, layers, '
                        'feature, precision (bits), time per run (ms), '
                        'performance (GFLOPs/sec), memory use (MB), comment\n')
                f = open('%s.csv'%logfile,'a+')
                f.write(header)
                f.close()

            if use_gpu:
                mem =  bm_conv.get_memory_use()
            else:
                mem = 0
            with open('%s.csv'%logfile,'a+') as f:
                f.write(conv.generate_logtext(timeUsed, ops, mem))

        if not args.no_timeline:
            bm_conv.run_timeline(
                    '%s_%dx%dx%dx%d' %(logfile,
                                       args.matsize,
                                       args.kernelsize,
                                       args.channels_in,
                                       args.channels_out),
                    args.batchsize)

        print("\n========================================\n\n")


    ########## Benchmark training step of VGG16 ##########
    if args.testVGG16:
        if args.logfile == '':
            logfile = str('/results/benchmark_VGG16_%s_%s'
                    %(args.device, time.strftime("%Y%m%d")))
        else:
            logfile = args.logfile

        model = benchmark_VGG16.VGG16(args)

        train_op, vgg16_graph = model.create_benchmark_op()

        bm_vgg16 = run_benchmark.benchmark(
                train_op,
                args.iter_warmup,
                args.iter_benchmark,
                args.iter_timeline,
                vgg16_graph)


        print("========================================\n")
        print("Start training VGG-16")
        timeUsed = bm_vgg16.run_benchmark()

        print("\nTraining VGG-16 (%dx%d pixel, float%d, batchsize %d): "
                "%.3f ms per batch / %.3f images per sec)"
                % (args.imgsize,
                args.imgsize,
                args.precision,
                args.batchsize,
                timeUsed*1000,
                args.batchsize/timeUsed))

        if not args.no_saving:
            if not os.path.isfile('%s.csv'%logfile):
                header = ('operation, imsize, precision (bits), batchsize,'
                        'time per batch (ms), performance (img/sec), '
                        'memory use (MB), comment\n')
                f = open('%s.csv'%logfile,'a+')
                f.write(header)
                f.close()

            if use_gpu:
                mem = bm_vgg16.get_memory_use()
            else:
                mem = 0
            with open('%s.csv'%logfile,'a+') as f:
                f.write(model.generate_logtext(timeUsed, mem))

        if not args.no_timeline:
            bm_vgg16.run_timeline(logfile, args.batchsize)
        print("\n========================================\n\n")

if __name__ == '__main__':
    tf.app.run()
