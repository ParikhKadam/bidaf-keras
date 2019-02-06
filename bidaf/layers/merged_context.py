from keras.engine.topology import Layer
from keras import backend as K


class MergedContext(Layer):

    def __init__(self, **kwargs):
        super(MergedContext, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MergedContext, self).build(input_shape)

    def call(self, inputs):
        encoded_context, context_to_query_attention, query_to_context_attention = inputs
        element_wise_multiply1 = encoded_context * context_to_query_attention
        element_wise_multiply2 = encoded_context * query_to_context_attention
        concatenated_tensor = K.concatenate(
            [encoded_context, context_to_query_attention, element_wise_multiply1, element_wise_multiply2], axis=-1)
        return concatenated_tensor

    def compute_output_shape(self, input_shape):
        encoded_context_shape, _, _ = input_shape
        return encoded_context_shape[:-1] + (encoded_context_shape[-1] * 4, )

    def get_config(self):
        config = super().get_config()
        return config
