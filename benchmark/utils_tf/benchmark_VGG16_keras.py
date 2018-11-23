"""Benchmark the training of VGG16 convpolutional neural
network on synthetic data. Returns images per time as
result metric.
"""

import tensorflow as tf
from tensorflow.python.client import timeline
import keras
import numpy as np
import time

def benchmark_VGG16(
        imgwidth,
        imgheight,
        numclasses,
        optimizer,
        iterations,
        batchsize,
        precision,
        logfile,
        generate_timeline):

    # Generate synthetic data
    datatype = eval('np.float%d' %(precision))
    batch_data = np.zeros(
            [batchsize,imgwidth,imgheight,3],
            dtype=datatype)
    batch_label = np.zeros(
            [batchsize,numclasses],
            dtype=np.int16)


    run_metadata = tf.RunMetadata()

    # Define model
    model = keras.applications.VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=[imgwidth,imgheight,3],
            pooling=None,
            classes=numclasses)

    # Define optimizer
    if optimizer=='sgd':
        opt = keras.optimizers.SGD(lr=0.01)
    elif optimizer=='rmsprop':
        opt = keras.optimizers.rmsprop(lr=0.0001)

    # Compile model
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # Run warm-up
    model.train_on_batch(batch_data,batch_label)

    # Run benchmark
    t_start = time.time()
    for i in range(iterations):
        model.train_on_batch(batch_data,batch_label)
    dur = time.time()-t_start

    img_per_sec = (iterations*batchsize)/dur

    logtext = ('VGG-16, %d, %d, %d, %.3f\n'
            %(imgwidth,precision,batchsize,img_per_sec))
    with open('%s.csv'%logfile,'a+') as f:
        f.write(logtext)


    if generate_timeline:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        tensorboard = keras.callbacks.TensorBoard(log_dir='%s_tb'%logfile, histogram_freq=1, write_graph=True, write_images=False)
        model.compile(optimizer=opt, loss='categorical_crossentropy', options=run_options, run_metadata=run_metadata)
        model.train_on_batch(batch_data,batch_label,callbacks=[tensorboard])


        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('%s_timeline.json'%logfile, 'w') as f:
            f.write(chrome_trace)
    return img_per_sec
