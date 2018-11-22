import json
import prediction
from sklearn.externals import joblib


def new_model(input_dimension,input_size):
    """Creates new model
    Args:
        input_dimension:
        input_size:
    """
    m = model()
    m['layers'] = {}
    m['input'] = {}
    m['input']['dimension'] = input_dimension
    m['input']['size'] = input_size
    return m


def import_model(path):
    """Import model from path"""
    m = model()
    with open(path) as json_data:
        tmpdict = json.load(json_data)
    m['layers'] = tmpdict['layers']
    m['input'] = tmpdict['input']
    return m


class model(dict):
    """Class for model"""

    def save(self,path):
        """Save model to path"""
        with open(path, 'w') as json_file:
            json.dump(self, json_file, indent=4)


    def describe(self):
        """Prints a description of of the class instance"""
        print('%d layer network\n' %(len(self['layers'])))
        print('Input size %dx%dx%d\n'
              %(self['input']['size'],self['input']['size'],
                self['input']['dimension']))
        for layer in self['layers']:
            print('%s (%s), now %dx%d with %d channels'
                  %(self['layers'][layer]['name'],
                    self['layers'][layer]['type'],
                    self['layers'][layer]['output_size'],
                    self['layers'][layer]['output_size'],
                    self['layers'][layer]['channels_out']))


    def add_layer(self,layer_type,layer_name,**kwargs):
        """Adds a layer to the class instance
        Args:
            layer_type: Type of layer ('Convolution','Fully_connected' or 'Max_pool')
            layer_name: Name of layer (string)
        Layer type specific args:
            Convolution:
                kernelsize
                channels_out
                padding
                strides
                use_bias
                activation
            Max_pool:
                pool_size
                strides
                padding
            Fully_connected:
        """

        num_layers = len(self['layers'])
        if num_layers==0:
            input_dimension = self['input']['dimension']
            input_size = self['input']['size']
        else:
            input_dimension = self['layers'][num_layers]['channels_out']
            input_size = self['layers'][num_layers]['output_size']


        self['layers'][num_layers+1] = {}     # Create new layer
        self['layers'][num_layers+1]['name'] = layer_name
        self['layers'][num_layers+1]['type'] = layer_type

        if layer_type.lower()=='convolution':
            padding_reduction = ((kwargs['padding'].lower()=='valid')*(kwargs['kernelsize']-1))
            output_size = ((input_size - padding_reduction)/kwargs['strides'])

            self['layers'][num_layers+1]['matsize'] = input_size
            self['layers'][num_layers+1]['kernelsize'] = kwargs['kernelsize']
            self['layers'][num_layers+1]['channels_in'] = input_dimension
            self['layers'][num_layers+1]['channels_out'] = kwargs['channels_out']
            self['layers'][num_layers+1]['padding'] = kwargs['padding']
            self['layers'][num_layers+1]['strides'] = kwargs['strides']
            self['layers'][num_layers+1]['use_bias'] = kwargs['use_bias']
            self['layers'][num_layers+1]['activation'] = kwargs['activation']
            self['layers'][num_layers+1]['output_size'] = output_size

        if layer_type.lower()=='max_pool':
            padding_reduction = ((kwargs['padding'].lower()=='valid')*(kwargs['pool_size']-1))
            output_size = ((input_size - padding_reduction)/kwargs['strides'])

            self['layers'][num_layers+1]['pool_size'] = kwargs['pool_size']
            self['layers'][num_layers+1]['strides'] = kwargs['strides']
            self['layers'][num_layers+1]['padding'] = kwargs['padding']
            self['layers'][num_layers+1]['output_size'] = output_size
            self['layers'][num_layers+1]['channels_out'] = input_dimension

        print('%s (%s), now %dx%d with %d channels'
              %(layer_name, layer_type, output_size, output_size,
                self['layers'][num_layers+1]['channels_out']))


    def remove_last_layer(self):
        """Removes last layer of class instance"""
        num_layers = len(self['layers'])
        if num_layers>0:
            del self['layers'][num_layers]


    def predict(self,
                gpu_def,
                optimizer='SGD',
                batchsize=1,
                model_file='models/all/saved_model',
                scaler_file='models/all/scaler_Conv.save'):
        """Predicts execution time of class instance
        Args:
            gpu_def: json file with GPU definition
            batchsize, default 1
            model_file, default 'models/all/saved_model',
            scaler_file, default 'models/all/scaler_Conv.save'
        """

        scaler = joblib.load(scaler_file)

        with open(gpu_def) as json_data:
            gpu = json.load(json_data)

        layer,time = prediction.predict_walltime(
                self, model_file, scaler, batchsize, optimizer,
                gpu['bandwidth'], gpu['cores'], gpu['clock'])

        return layer, time
