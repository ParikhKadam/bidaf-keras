from keras.engine.topology import Layer
from keras.activations import linear
from keras import backend as K


class Similarity(Layer):

    def __init__(self, **kwargs):
        super(Similarity, self).__init__(**kwargs)

    def compute_similarity(self, repeated_context_vectors, repeated_query_vectors):
        element_wise_multiply = repeated_context_vectors * repeated_query_vectors
        concatenated_tensor = K.concatenate(
            [repeated_context_vectors, repeated_query_vectors, element_wise_multiply], axis=-1)
        dot_product = K.squeeze(K.dot(concatenated_tensor, self.kernel), axis=-1)
        return linear(dot_product + self.bias)

    def build(self, input_shape):
        word_vector_dim = input_shape[0][-1]
        weight_vector_dim = word_vector_dim * 3
        self.kernel = self.add_weight(name='similarity_weight',
                                      shape=(weight_vector_dim, 1),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='similarity_bias',
                                    shape=(),
                                    initializer='ones',
                                    trainable=True)
        super(Similarity, self).build(input_shape)

    def call(self, inputs):
        context_vectors, query_vectors = inputs
        num_context_words = K.shape(context_vectors)[1]
        num_query_words = K.shape(query_vectors)[1]
        context_dim_repeat = K.concatenate([[1, 1], [num_query_words], [1]], 0)
        query_dim_repeat = K.concatenate([[1], [num_context_words], [1, 1]], 0)
        repeated_context_vectors = K.tile(K.expand_dims(context_vectors, axis=2), context_dim_repeat)
        repeated_query_vectors = K.tile(K.expand_dims(query_vectors, axis=1), query_dim_repeat)
        similarity_matrix = self.compute_similarity(repeated_context_vectors, repeated_query_vectors)
        return similarity_matrix

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        num_context_words = input_shape[0][1]
        num_query_words = input_shape[1][1]
        return (batch_size, num_context_words, num_query_words)

    def get_config(self):
        config = super().get_config()
        return config
