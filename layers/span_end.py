from keras.engine.topology import Layer
from keras.layers.advanced_activations import Softmax
from keras.layers import TimeDistributed, Dense, LSTM, Bidirectional
from keras import backend as K


class SpanEnd(Layer):

    def __init__(self, **kwargs):
        super(SpanEnd, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SpanEnd, self).build(input_shape)

    def call(self, inputs):
        encoded_passage, merged_context, modeled_passage, span_begin_probabilities = inputs
        passage_weighted_by_predicted_span = K.tile(
            K.sum(K.expand_dims(span_begin_probabilities, axis=-1) * modeled_passage, -2),
            [1, K.shape(encoded_passage)[1], 1])
        multiply1 = modeled_passage * passage_weighted_by_predicted_span
        span_end_representation = K.concatenate(
            [merged_context, modeled_passage, passage_weighted_by_predicted_span, multiply1])
        emdim = K.int_shape(encoded_passage)[-1] // 2
        span_end_representation = Bidirectional(LSTM(emdim, return_sequences=True))(span_end_representation)
        span_end_input = K.concatenate([merged_context, span_end_representation])
        span_end_weights = TimeDistributed(Dense(units=1))(span_end_input)
        span_end_probabilities = Softmax()(span_end_weights)
        return span_end_probabilities

    def compute_output_shape(self, input_shape):
        _, merged_context_shape, _, _ = input_shape
        return merged_context_shape[:-1]
