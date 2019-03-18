from keras.engine.topology import Layer
from keras.layers.advanced_activations import Softmax
from keras.layers import TimeDistributed, Dense, LSTM, Bidirectional
from keras import backend as K


class SpanEnd(Layer):

    def __init__(self, **kwargs):
        super(SpanEnd, self).__init__(**kwargs)

    def build(self, input_shape):
        emdim = input_shape[0][-1] // 2
        input_shape_bilstm_1 = input_shape[0][:-1] + (emdim*14, )
        self.bilstm_1 = Bidirectional(LSTM(emdim, return_sequences=True))
        self.bilstm_1.build(input_shape_bilstm_1)
        input_shape_dense_1 = input_shape[0][:-1] + (emdim*10, )
        self.dense_1 = Dense(units=1)
        self.dense_1.build(input_shape_dense_1)
        self.trainable_weights = self.bilstm_1.trainable_weights + self.dense_1.trainable_weights
        super(SpanEnd, self).build(input_shape)

    def call(self, inputs):
        encoded_passage, merged_context, modeled_passage, span_begin_probabilities = inputs
        weighted_sum = K.sum(K.expand_dims(span_begin_probabilities, axis=-1) * modeled_passage, -2)
        passage_weighted_by_predicted_span = K.expand_dims(weighted_sum, axis=1)
        tile_shape = K.concatenate([[1], [K.shape(encoded_passage)[1]], [1]], axis=0)
        passage_weighted_by_predicted_span = K.tile(passage_weighted_by_predicted_span, tile_shape)
        multiply1 = modeled_passage * passage_weighted_by_predicted_span
        span_end_representation = K.concatenate(
            [merged_context, modeled_passage, passage_weighted_by_predicted_span, multiply1])

        span_end_representation = self.bilstm_1(span_end_representation)

        span_end_input = K.concatenate([merged_context, span_end_representation])

        span_end_weights = TimeDistributed(self.dense_1)(span_end_input)

        span_end_probabilities = Softmax()(K.squeeze(span_end_weights, axis=-1))
        return span_end_probabilities

    def compute_output_shape(self, input_shape):
        _, merged_context_shape, _, _ = input_shape
        return merged_context_shape[:-1]

    def get_config(self):
        config = super().get_config()
        return config
