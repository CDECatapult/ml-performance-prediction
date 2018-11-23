#!/bin/bash
batchsize=(1 2 4 8 16 32 64)

for i in {0..6};
    do python benchmark.py --testVGG16 --no_timeline --iter_benchmark=100 --batchsize=${batchsize[i]};
done
