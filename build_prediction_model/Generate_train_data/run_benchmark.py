import tensorflow as tf
import time


class benchmark(object):
    """Class for running the benchmarks"""

    def __init__(self, benchmark_op, iterations_warmup,
                 iterations_benchmark, graph):
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
        self.graph = graph
        self.config = tf.ConfigProto(
                graph_options=tf.GraphOptions(
                        optimizer_options=tf.OptimizerOptions(
                                opt_level=tf.OptimizerOptions.L0)),
                log_device_placement=False)


    def run_benchmark(self):
        """Run benchmark, return time per iteration in ms"""
        with tf.Session(config=self.config, graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Warm-up run
            for _ in range(self.iterations_warmup):
                sess.run(self.benchmark_op)

            # Benchmark run
            t = time.time()
            for _ in range(self.iterations_benchmark):
                sess.run(self.benchmark_op)
            timeUsed = (time.time()-t)/self.iterations_benchmark * 1000
        return timeUsed
