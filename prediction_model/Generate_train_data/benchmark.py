"""Generates benchmarks for convolutions with different, randomly determined
parameters. Saves results into a pandas dataframe (.pkl)
and a numpy array (.npy)
"""
import os
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.stats as stats
import time
import benchmark_conv, benchmark_dense
import run_benchmark


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


parser = argparse.ArgumentParser('Benchmarking convolutions')

# Benchmarks to perform
parser.add_argument('--testDense', action="store_true", default=False,
                    help='Benchmark fully connected layer/matrix multiplication')
parser.add_argument('--testConv', action="store_true", default=False,
                    help='Benchmark 2D convolution')
# General parameters
parser.add_argument('--backprop_ratio', type=float, default=0.5,
                    help='ratio of iterations with backward pass ([0..1])')
parser.add_argument('--num_gpu', type=int, default=1,
                    help='Number of GPUs to use')
parser.add_argument('--devlist', type=str, default='',
                    help='List of devices to use, overwrites num_gpu if set')
parser.add_argument('--num_val', type=int, default=100000,
                    help='Number of results to compute')
parser.add_argument('--logfile', type=str, default='',
                    help='Text file to store results')
parser.add_argument('--device', type=str, default='',
                    help='Device name as appearing in logfile')
parser.add_argument('--iter_benchmark', type=int, default=50,
                    help='Number of iterations for benchmark')
parser.add_argument('--iter_warmup', type=int, default=5,
                    help='Number of iterations for warm-up')
parser.add_argument('--repetitions', type=int, default=5,
                    help='Number of repetitions of the same experiment')

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

    optimizer_list = [
            'None',
            'GradientDescentOptimizer(learning_rate=0.0001)',
            'AdadeltaOptimizer(learning_rate=0.0001)',
            'AdagradOptimizer(learning_rate=0.0001)',
            'MomentumOptimizer(learning_rate=0.0001,momentum=0.1)',
            'AdamOptimizer(learning_rate=0.0001)',
            'RMSPropOptimizer(learning_rate=0.0001)']

    activation_list = [
            'None',
            'tf.nn.relu',
            'tf.nn.tanh',
            'tf.nn.sigmoid']


    ########### Benchmark convolution ##########
    if args.testConv:
        if args.logfile == '':
                logfile = str('/results/benchmark_convolution_%s_%s'
                        %(args.device, time.strftime("%Y%m%d")))
        else:
            logfile = args.logfile

        # Set random parameters
        batchsize = np.random.randint(1,65,args.num_val)
        matsize = np.random.randint(1,513,args.num_val)
        kernelsize = np.zeros(args.num_val,dtype=np.int32)
        channels_in = np.zeros(args.num_val,dtype=np.int32)
        channels_out = np.zeros(args.num_val,dtype=np.int32)
        strides = np.random.randint(1,5,args.num_val)
        optimizer = np.zeros(args.num_val,dtype=np.int32)
        precision = (np.ones(args.num_val)*32).astype(int) # np.random.choice([16,32],args.num_val)
        padding = np.random.randint(0,2,args.num_val)
        activation_fct = np.random.randint(0,4,args.num_val)
        use_bias = np.random.choice([True,False],args.num_val)

        gpu_index = np.arange(args.num_val)%(len(devlist))

        timeUsed = np.zeros([args.num_val,args.repetitions])

        tprint = time.time()
        for i in range(args.num_val):
            kernelsize[i] = np.random.randint(1,min(7,matsize[i])+1)
            channels_in[i] = np.random.randint(1,10000/matsize[i])
            channels_out[i] = np.random.randint(1,10000/matsize[i])
            if np.random.rand()<=args.backprop_ratio:
                optimizer[i] = np.random.randint(1,len(optimizer_list))
            else:
                optimizer[i] = 0

        # Run benchmarks
        for rep in range(args.repetitions):
            print("Benchmarking convolution, starting repetition %d" %rep)
            for i in range(args.num_val):
                conv = benchmark_conv.convolution(
                        batchsize[i],
                        matsize[i],
                        kernelsize[i],
                        channels_in[i],
                        channels_out[i],
                        strides[i],
                        precision[i],
                        ('SAME' if padding[i]==1 else 'VALID'),
                        activation_list[activation_fct[i]],
                        use_bias[i],
                        devlist,
                        optimizer_list[optimizer[i]])

                if optimizer[i]==0:
                    benchmark_op, benchmark_graph = conv.create_benchmark_op()
                else:
                    benchmark_op, benchmark_graph = conv.create_benchmark_op_with_backprop()

                bm_conv = run_benchmark.benchmark(
                        benchmark_op,
                        args.iter_warmup,
                        args.iter_benchmark,
                        benchmark_graph)

                try:
                    timeUsed[i,rep] =  bm_conv.run_benchmark()
                except:
                    print('Error: Out of GPU memory')
                    timeUsed[i,rep] = None

                if (i+1)%100==0:
                    print("Iteration %d / %d: Finished convolution %d / %d "
                            "(%.2f sec): t = %.3f ms \n"
                            %(rep+1,
                            args.repetitions,
                            i+1,
                            args.num_val,
                            time.time()-tprint,
                            timeUsed[i,rep]))

        # Generate dataframe and save results
        print("Generating dataframe and saving results")
        df_results = pd.DataFrame({
                'batchsize': batchsize,
                'matsize': matsize,
                'kernelsize': kernelsize,
                'channels_in': channels_in,
                'channels_out': channels_out,
                'strides': strides,
                'padding': padding,
                'precision': precision,
                'activation_fct': activation_fct,
                'use_bias': use_bias,
                'optimizer': optimizer,
                'gpu': gpu_index,
                'timeUsed_median': np.median(timeUsed,1),
                'timeUsed_min': np.min(timeUsed,1),
                'timeUsed_max': np.max(timeUsed,1),
                'timeUsed_std': np.std(timeUsed,1)})

        df_results.to_pickle('%s.pkl' %logfile)
        np.save('%s.npy' %logfile, timeUsed)


    ########### Benchmark fully connected layer ##########
    if args.testDense:
        if args.logfile == '':
                logfile = str('/results/benchmark_dense_%s_%s'
                        %(args.device, time.strftime("%Y%m%d")))
        else:
            logfile = args.logfile

        # Set random parameters
        batchsize = np.random.randint(1,65,args.num_val)
        dim_input = np.random.randint(1,4096,args.num_val)
        dim_output = np.random.randint(1,4096,args.num_val)
        precision = (np.ones(args.num_val)*32).astype(int) # np.random.choice([16,32],args.num_val)
        activation_fct = np.random.randint(0,4,args.num_val)
        optimizer = np.zeros(args.num_val,dtype=np.int32)
        gpu_index = np.arange(args.num_val)%(len(devlist))

        timeUsed = np.zeros([args.num_val,args.repetitions])

        tprint = time.time()

        for i in range(args.num_val):
            if np.random.rand()<=args.backprop_ratio:
                optimizer[i] = np.random.randint(1,len(optimizer_list))
            else:
                optimizer[i] = 0

        # Run benchmarks
        for rep in range(args.repetitions):
            print("Benchmarking fully connected, starting repetition %d" %rep)
            for i in range(args.num_val):
                dense = benchmark_dense.dense_layer(
                        dim_input[i],
                        dim_output[i],
                        batchsize[i],
                        precision[i],
                        activation_list[activation_fct[i]],
                        optimizer_list[optimizer[i]],
                        devlist)

                if optimizer[i]==0:
                    benchmark_op, benchmark_graph = dense.create_benchmark_op()
                else:
                    benchmark_op, benchmark_graph = dense.create_benchmark_op_with_backprop()

                bm_dense = run_benchmark.benchmark(
                        benchmark_op,
                        args.iter_warmup,
                        args.iter_benchmark,
                        benchmark_graph)

                try:
                    timeUsed[i,rep] =  bm_dense.run_benchmark()
                except:
                    print('Error: Out of GPU memory')
                    timeUsed[i,rep] = None

                if (i+1)%100==0:
                    print("Iteration %d / %d: Finished dense layer %d / %d "
                            "(%.2f sec): t = %.3f ms \n"
                            %(rep+1,
                            args.repetitions,
                            i+1,
                            args.num_val,
                            time.time()-tprint,
                            timeUsed[i,rep]))

        # Generate dataframe and save results
        print("Generating dataframe and saving results")
        df_results = pd.DataFrame({
                'batchsize': batchsize,
                'dim_input': dim_input,
                'dim_output': dim_output,
                'precision': precision,
                'activation_fct': activation_fct,
                'optimizer': optimizer,
                'gpu': gpu_index,
                'timeUsed_median': np.median(timeUsed,1),
                'timeUsed_min': np.min(timeUsed,1),
                'timeUsed_max': np.max(timeUsed,1),
                'timeUsed_std': np.std(timeUsed,1)})

        df_results.to_pickle('%s.pkl' %logfile)
        np.save('%s.npy' %logfile, timeUsed)


if __name__ == '__main__':
    tf.app.run()
