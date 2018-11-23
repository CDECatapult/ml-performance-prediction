import tensorflow as tf
import time
from tensorflow.python.client import timeline
from utils_tf import merge_timeline


class benchmark(object):
    """Class for running the benchmarks"""

    def __init__(self, benchmark_op, iterations_warmup,
                 iterations_benchmark, iterations_timeline, graph):
        """Initialize benchmark

        Args:
            benchmark_op: tf tensor, operation to be executed in benchmark
            iterations_warmup: Number of iterations for warm-up
            iterations_benchmark: Number of iterations for benchmark
            iterations_timeline: Number of iterations for generation of timeline
            graph: tf graph
        """

        self.benchmark_op = benchmark_op
        self.iterations_warmup = iterations_warmup
        self.iterations_benchmark = iterations_benchmark
        self.iterations_timeline = iterations_timeline
        self.graph = graph
        self.config = tf.ConfigProto(
                graph_options=tf.GraphOptions(
                        optimizer_options=tf.OptimizerOptions(
                                opt_level=tf.OptimizerOptions.L0)),
                log_device_placement=False)


    def run_benchmark(self):
        """Run benchmark"""
        with tf.Session(config=self.config, graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Warm-up run
            for _ in range(self.iterations_warmup):
                sess.run(self.benchmark_op)

            # Benchmark run
            t = time.time()
            for _ in range(self.iterations_benchmark):
                sess.run(self.benchmark_op)
            timeUsed = (time.time()-t)/self.iterations_benchmark
        return timeUsed


    def run_timeline(self, logfile, batchsize):
        """Run benchmark with generation of timeline"""
        run_metadata = tf.RunMetadata()
        with tf.Session(config=self.config, graph=self.graph) as sess:
            train_writer = tf.summary.FileWriter('%s_tb' % logfile, self.graph)
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            sess.run(tf.global_variables_initializer())

            # Run warm-up
            for _ in range(self.iterations_warmup):
                sess.run(self.benchmark_op, options=options, run_metadata=run_metadata)

            # Benchmark run
            many_runs_timeline = merge_timeline.TimeLiner()
            t_start = time.time()
            for _ in range(self.iterations_timeline):
                sess.run(self.benchmark_op, options=options, run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                many_runs_timeline.update_timeline(chrome_trace)

            print("Time for timeline run: %.3f ms"
                    %((time.time()-t_start)*1000/self.iterations_timeline))

            many_runs_timeline.save('%s_%dbatch_timeline.json' % (logfile,batchsize))

            train_writer.add_run_metadata(run_metadata,'single convolution')
            train_writer.close()
        return


    def get_memory_use(self):
        """Evaluates memory usage"""
        with tf.Session(config=self.config, graph=self.graph) as sess:
            mem = sess.run(tf.contrib.memory_stats.MaxBytesInUse())
        return mem
