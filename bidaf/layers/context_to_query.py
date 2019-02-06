from keras.engine.topology import Layer
from keras.layers.advanced_activations import Softmax
from keras import backend as K


class C2QAttention(Layer):

    def __init__(self, **kwargs):
        super(C2QAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(C2QAttention, self).build(input_shape)

    def call(self, inputs):
        similarity_matrix, encoded_question = inputs
        context_to_query_attention = Softmax(axis=-1)(similarity_matrix)
        encoded_question = K.expand_dims(encoded_question, axis=1)
        return K.sum(K.expand_dims(context_to_query_attention, axis=-1) * encoded_question, -2)

    def compute_output_shape(self, input_shape):
        similarity_matrix_shape, encoded_question_shape = input_shape
        return similarity_matrix_shape[:-1] + encoded_question_shape[-1:]

    def get_config(self):
        config = super().get_config()
        return config
