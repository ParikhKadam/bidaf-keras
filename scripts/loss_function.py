from keras import backend as K


def negative_avg_log_error(y_true, y_pred):

    def sum_of_log_probabilities(true_and_pred):
        y_true, y_pred_start, y_pred_end = true_and_pred

        start_probability = y_pred_start[K.cast(y_true[0], dtype='int32')]
        end_probability = y_pred_end[K.cast(y_true[1], dtype='int32')]
        return K.log(start_probability) + K.log(end_probability)

    y_true = K.squeeze(y_true, axis=1)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]
    batch_probability_sum = K.map_fn(sum_of_log_probabilities, (y_true, y_pred_start, y_pred_end), dtype='float32')
    return -K.mean(batch_probability_sum, axis=0)
