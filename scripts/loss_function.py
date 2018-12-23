from keras import backend as K


def negative_avg_log_error(y_true, y_pred):

    def sum_of_log_probabilities(true_and_pred):
        y_true = true_and_pred[0]
        y_pred = true_and_pred[1]
        print(K.int_shape(y_true))
        print(K.int_shape(y_pred))
        start_index = int(y_true[0])
        end_index = int(y_true[1])
        start_probability = y_pred[start_index]
        end_probability = y_pred[end_index]
        return K.log(start_probability) + K.log(end_probability)

    print(K.int_shape(y_true))
    print(K.int_shape(y_pred))
    batch_probability_sum = K.map_fn(lambda x: sum_of_log_probabilities(x), elems=(y_true, y_pred), dtype='float32')
    return -K.mean(batch_probability_sum, axis=1)
