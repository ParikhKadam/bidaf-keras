from keras.engine.topology import Layer
from keras import backend as K


class CombineOutputs(Layer):

    def __init__(self, **kwargs):
        super(CombineOutputs, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CombineOutputs, self).build(input_shape)

    def call(self, inputs):
        span_begin_probabilities, span_end_probabilities = inputs
        return K.stack([span_begin_probabilities, span_end_probabilities],axis = 1)\

    def compute_output_shape(self, input_shape):
        number_of_tensors = len(input_shape)
        return input_shape[0][0:1] + (number_of_tensors, ) + input_shape[0][1:]

    def get_config(self):
        config = super().get_config()
        return config
