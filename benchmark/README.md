# MLbenchmark
A toolkit for benchmarking ML applications on different hardware.

Run
```bash
python benchmark.py
```
with the following optional arguments:

#### Types of benchmarks
--testMatMul (Benchmark matrix multiplication)<br/>
--testConv (Benchmark 2D convolution)<br/>
--testVGG16 (Benchmark training a VGG16 cnn on sythetic data)<br/>

#### General parameters
--num_gpu (Number of GPUs to use, default 1)<br/>
--devlist (List of devices to use, overwrites num_gpu if set, default '')<br/>
--precision (Precision of floats used as data, default 32)<br/>
--logfile (Text file to store results, leave empty to automatically generate, default '')<br/>
--device (Device name that appears in automatically generated logfile)<br/>
--use_tf_profiler (Calculate number of FLOPs with ft profiler, default FALSE)
--generate_timeline (Make tf timeline, default FALSE)

#### Parameters for matrix multiplication / convolution
--iter (Number of iterations, default 10)<br/>
--logFLOPs (log10 of number of FLOPs to perform (will be rounded up), overwrites iter, set 0 or leave empty use iter, default 0)<br/>
--matsize (Size of each matrix for benchmark, default 1024)<br/>
--kernelsize (Size of kernel for benchmarking convolution, default 15)<br/>
--channels_in (Number of layers going into a convolution, default 1)
--channels_out (Number of features going into a convolution, default 1)

#### Parameters for CNNs
--imgsize (Size of (square) images, default 50)<br/>
--batchsize (Batch size for training CNN, default 128)<br/>
--numclasses (Number of image classes (labels), default 1000)<br/>
--optimizer (Optimzer used for VGG-16, must be one of sgd or rmsprop, default sgd'<br/>


#### Build and run a docker container:
```bash
sudo docker build -t mlbenchmark .
sudo docker run --runtime=nvidia -it --rm --mount type=bind,source=...,target=/results mlbenchmark
```

#### Example for matrix-matrix multiplication benchmark:
```bash
lp=(16 32)
ls=(1024 2048 4096 8192 16384)
for p in ${lp[*]}; do
    for s in ${ls[*]}; do           
        python benchmark.py --testMatMul --logFLOPs=15 --precision=$p --matsize=$s
    done
done
```
