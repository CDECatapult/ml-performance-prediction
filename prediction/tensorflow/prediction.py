import json
import tensorflow as tf
import numpy as np
from sklearn.externals import joblib


def predict_walltime(model,
                     model_file,
                     scaler,
                     batchsize,
                     optimizer,
                     bandwidth,
                     cores,
                     clock):

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["serve"], model_file)
        graph = tf.get_default_graph()

        layer_prediction = []
        layer_name = []

        for layer in model['layers']:
            if model['layers'][layer]['type']=='Convolution':
                features = get_input_features(model['layers'][layer],
                                              scaler,
                                              batchsize,
                                              optimizer,
                                              bandwidth,
                                              cores,
                                              clock)

                run = sess.run(
                        'model_prediction:0',
                        feed_dict={'model_input:0': features,
                                   'model_istraining:0': False})
                layer_prediction.append(run[0])
                layer_name.append(model['layers'][layer]['name'])

    return layer_name, layer_prediction


def get_input_features(dictionary,scaler,batchsize,optimizer,bandwidth,cores,clock):

    padding_reduction = ((dictionary['padding'].lower()=='valid')
                         *(dictionary['kernelsize']-1))
    elements_output = ((dictionary['matsize'] - padding_reduction)
                       / dictionary['strides'])**2

    ops = (batchsize
           * elements_output
           * dictionary['kernelsize']**2
           * dictionary['channels_in']
           * dictionary['channels_out'])

    memory_weights = (dictionary['kernelsize']**2
                      * dictionary['channels_in']
                      * dictionary['channels_out']
                      + dictionary['use_bias'] * dictionary['channels_out'])

    memory_in = (batchsize
                 * dictionary['matsize']**2
                 * dictionary['channels_in'])
    memory_out = (batchsize
                  * elements_output
                  * dictionary['channels_out'])

    features = np.array([batchsize,
                         dictionary['matsize']**2,
                         dictionary['kernelsize']**2,
                         dictionary['channels_in'],
                         dictionary['channels_out'],
                         (1 if dictionary['padding'].lower()=='same' else 0),
                         dictionary['strides'],
                         dictionary['use_bias'],
                         (1 if optimizer.lower()=='sgd' else 0),
                         (1 if optimizer.lower()=='adadelta' else 0),
                         (1 if optimizer.lower()=='adagrad' else 0),
                         (1 if optimizer.lower()=='momentum' else 0),
                         (1 if optimizer.lower()=='adam' else 0),
                         (1 if optimizer.lower()=='rmsprop' else 0),
                         (1 if dictionary['activation'].lower()=='relu' else 0),
                         (1 if dictionary['activation'].lower()=='tanh' else 0),
                         (1 if dictionary['activation'].lower()=='sigmoid' else 0),
                         # ops,
                         # memory_weights,
                         # memory_in,
                         # memory_out,
                         bandwidth,
                         cores,
                         clock
                         ])

    features = scaler.transform(features.reshape(1, -1))
    return features
