from keras.engine.topology import Layer
from keras.layers.advanced_activations import Softmax
from keras.layers import TimeDistributed, Dense
from keras import backend as K


class SpanBegin(Layer):

    def __init__(self, **kwargs):
        super(SpanBegin, self).__init__(**kwargs)

    def build(self, input_shape):
        last_dim = input_shape[0][-1] + input_shape[1][-1]
        input_shape_dense_1 = input_shape[0][:-1] + (last_dim, )
        self.dense_1 = Dense(units=1)
        self.dense_1.build(input_shape_dense_1)
        self.trainable_weights = self.dense_1.trainable_weights
        super(SpanBegin, self).build(input_shape)

    def call(self, inputs):
        merged_context, modeled_passage = inputs
        span_begin_input = K.concatenate([merged_context, modeled_passage])
        span_begin_weights = TimeDistributed(self.dense_1)(span_begin_input)
        span_begin_probabilities = Softmax()(K.squeeze(span_begin_weights, axis=-1))
        return span_begin_probabilities

    def compute_output_shape(self, input_shape):
        merged_context_shape, _ = input_shape
        return merged_context_shape[:-1]

    def get_config(self):
        config = super().get_config()
        return config
