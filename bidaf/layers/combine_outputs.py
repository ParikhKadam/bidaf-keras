from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import Lambda, Concatenate


class CombineOutputs(Layer):

    def __init__(self, **kwargs):
        super(CombineOutputs, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CombineOutputs, self).build(input_shape)

    def call(self, inputs):
        span_begin_probabilities, span_end_probabilities = inputs
        # return K.stack([span_begin_probabilities, span_end_probabilities],axis = 1)\
        layer1 = Lambda(lambda x: K.expand_dims(span_begin_probabilities, axis=1))(span_begin_probabilities)
        layer2 = Lambda(lambda x: K.expand_dims(span_end_probabilities, axis=1))(span_end_probabilities)
        concat_layer = Concatenate(axis=1)([layer1, layer2])
        return concat_layer

    def compute_output_shape(self, input_shape):
        number_of_tensors = len(input_shape)
        return input_shape[0][0:1] + (number_of_tensors, ) + input_shape[0][1:]

    def get_config(self):
        config = super().get_config()
        return config
