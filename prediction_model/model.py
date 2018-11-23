import tensorflow as tf
import numpy as np
import os

class Model:
    """Model class"""
    def __init__(self,inputs,targets,learning_rate,reg_constant,dropout_rate,
                 num_neurons,lr_initial,lr_decay_step,batch_size,model_name):

        self.inputs = inputs
        self.targets = targets
        self.learning_rate = learning_rate
        self.reg_constant = reg_constant
        self.dropout_rate = dropout_rate
        self.num_layers = len(num_neurons)
        self.num_neurons = num_neurons
        self.lr_initial = lr_initial
        self.lr_decay_step = lr_decay_step
        self.batch_size = batch_size
        self.model_name = model_name

        self._prediction = None
        self._loss = None
        self._train_op = None
        self._summary_op = None

        self.istraining = tf.placeholder(tf.bool,
                                         shape=None,
                                         name='model_istraining')

        self.global_step = tf.get_variable('global_step',
                                           initializer=tf.constant(0),
                                           trainable=False)


    @property
    def prediction(self):
        """Funtion to build the model and generate a prediction
        returns property
        """
        if self._prediction is None:
            data_dim = self.inputs.shape[1]

            fcLayer = self.inputs
            # Some fully connected layers
            for layer in range(0,self.num_layers):
                name_scope = 'layer_fc%d' %layer
                with tf.variable_scope(name_scope) as scope:
                    fcLayer = tf.layers.dense(
                            inputs = fcLayer,
                            units = self.num_neurons[layer],
                            activation = tf.nn.relu,
                            kernel_regularizer = tf.contrib.layers.l2_regularizer(self.reg_constant),
                            use_bias = True)

            # Dropout only after last layer
            fcLayer = tf.layers.dropout(inputs=fcLayer,
                                        rate=self.dropout_rate,
                                        training=self.istraining)

            # Prediction. Relu for preventing negative results
            output = tf.layers.dense(
                    inputs = fcLayer,
                    units = 1,
                    activation = tf.nn.relu,
                    use_bias = False)

            self._prediction = tf.reshape(output, [-1], name='model_prediction')
        return self._prediction


    @property
    def loss(self):
        """Function that generates the loss, returns property"""
        if self._loss is None:
            self._loss = (tf.losses.get_regularization_loss()
                         + tf.losses.mean_squared_error(
                               labels=tf.log(1+self.targets),
                               predictions=tf.log(1+self.prediction)))

            # self._loss = (tf.losses.get_regularization_loss()
            #               + tf.losses.mean_squared_error(
            #                     labels=self.targets,
            #                     predictions=self.prediction))
        return self._loss


    @property
    def train_op(self):
        """Function that generates the train operation, returns property"""
        if self._train_op is None:
            opt = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate,
                    epsilon=.1)
            self._train_op = opt.minimize(
                    self.loss,
                    global_step=self.global_step)
        return self._train_op


    @property
    def summary_op(self):
        """Function to write the summary, returns property"""
        if self._summary_op is None:
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("learning_rate", self.learning_rate)
            tf.summary.histogram("histogram_loss", self.loss)
            self._summary_op =  tf.summary.merge_all()
        return self._summary_op


    def train(self, traindata, trainlabel, testdata, testlabel, num_train_steps):
        """Train the model for a number of steps, save checkpoints, write
        graph and loss for tensorboard
        Inputs:
            traindata
            trainlabel
            testdata
            testlabel
            num_train_steps
        """

        print(os.path.abspath('./'))
        saver = tf.train.Saver()

        initial_step = 0

        try:
            os.mkdir('./checkpoints/%s' %self.model_name)
        except:
            pass

        num_datapoints = traindata.shape[0]
        list_datapoints = np.arange(0,num_datapoints)
        num_batches = np.int(np.ceil(num_datapoints/self.batch_size))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoints/%s/checkpoint' %self.model_name))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            writer_train = tf.summary.FileWriter('./graphs/prediction/train/%s' %self.model_name, sess.graph)
            writer_test = tf.summary.FileWriter('./graphs/prediction/test/%s' %self.model_name, sess.graph)

            initial_step = self.global_step.eval()

            for epoch in range(initial_step, initial_step + num_train_steps):
                np.random.shuffle(list_datapoints)
                avg_loss = 0
                for i in range(0,num_batches):
                    batch = list_datapoints[i*self.batch_size:min((i+1)*self.batch_size,num_datapoints)]
                    _, loss, summary = sess.run(
                            [self.train_op, self.loss, self.summary_op],
                            feed_dict={
                                    self.inputs: traindata[batch,:],
                                    self.targets: trainlabel[batch],
                                    self.learning_rate: self.lr_initial*2**(-np.floor(epoch/self.lr_decay_step)),
                                    self.istraining: True})
                    avg_loss += loss/num_batches
                writer_train.add_summary(summary, global_step=epoch)

                testloss, testsummary = sess.run(
                        [self.loss,self.summary_op],
                        feed_dict={
                                self.inputs: testdata,
                                self.targets: testlabel,
                                self.learning_rate: self.lr_initial*2**(-np.floor(i/self.lr_decay_step)),
                                self.istraining: False})
                writer_test.add_summary(testsummary, global_step=epoch)
                saver.save(sess, './checkpoints/%s/prediction' %self.model_name, epoch)
                if epoch%10==0:
                    print('Epoch {}: Train loss {:.3f}, Test loss {:.3f}'.format(epoch, avg_loss, testloss))


            writer_train.close()
            writer_test.close()
