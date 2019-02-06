from keras.engine.topology import Layer
from keras.layers.advanced_activations import Softmax
from keras import backend as K


class Q2CAttention(Layer):

    def __init__(self, **kwargs):
        super(Q2CAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Q2CAttention, self).build(input_shape)

    def call(self, inputs):
        similarity_matrix, encoded_context = inputs
        max_similarity = K.max(similarity_matrix, axis=-1)
        # by default, axis = -1 in Softmax
        context_to_query_attention = Softmax()(max_similarity)
        weighted_sum = K.sum(K.expand_dims(context_to_query_attention, axis=-1) * encoded_context, -2)
        expanded_weighted_sum = K.expand_dims(weighted_sum, 1)
        num_of_repeatations = K.shape(encoded_context)[1]
        return K.tile(expanded_weighted_sum, [1, num_of_repeatations, 1])

    def compute_output_shape(self, input_shape):
        similarity_matrix_shape, encoded_context_shape = input_shape
        return similarity_matrix_shape[:-1] + encoded_context_shape[-1:]

    def get_config(self):
        config = super().get_config()
        return config
