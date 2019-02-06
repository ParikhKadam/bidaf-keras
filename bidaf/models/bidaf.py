from keras.layers import Input, TimeDistributed, LSTM, Bidirectional
from keras.models import Model
from keras.optimizers import Adadelta
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils import multi_gpu_model
from ..layers import Highway, Similarity, C2QAttention, Q2CAttention, MergedContext, SpanBegin, SpanEnd, CombineOutputs
from ..scripts import negative_avg_log_error, accuracy
import os


class BidirectionalAttentionFlow():

    def __init__(self, emdim, num_highway_layers=2, num_decoders=1, encoder_dropout=0, decoder_dropout=0):
        self.emdim = emdim

        question_input = Input(shape=(None, emdim), dtype='float32', name="question_input")
        passage_input = Input(shape=(None, emdim), dtype='float32', name="passage_input")

        for i in range(num_highway_layers):
            highway_layer = Highway(name='highway_{}'.format(i))
            question_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_qtd")
            question_embedding = question_layer(question_input)
            passage_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_ptd")
            passage_embedding = passage_layer(passage_input)

        encoder_layer = Bidirectional(LSTM(emdim, recurrent_dropout=encoder_dropout,
                                           return_sequences=True), name='bidirectional_encoder')
        encoded_question = encoder_layer(question_embedding)
        encoded_passage = encoder_layer(passage_embedding)

        similarity_matrix = Similarity(name='similarity_layer')([encoded_passage, encoded_question])

        context_to_query_attention = C2QAttention(name='context_to_query_attention')([
            similarity_matrix, encoded_question])
        query_to_context_attention = Q2CAttention(name='query_to_context_attention')([
            similarity_matrix, encoded_passage])

        merged_context = MergedContext(name='merged_context')(
            [encoded_passage, context_to_query_attention, query_to_context_attention])

        modeled_passage = merged_context
        for i in range(num_decoders):
            hidden_layer = Bidirectional(LSTM(emdim, recurrent_dropout=decoder_dropout,
                                              return_sequences=True), name='bidirectional_decoder_{}'.format(i))
            modeled_passage = hidden_layer(modeled_passage)

        span_begin_probabilities = SpanBegin(name='span_begin')([merged_context, modeled_passage])
        span_end_probabilities = SpanEnd(name='span_end')(
            [encoded_passage, merged_context, modeled_passage, span_begin_probabilities])

        output = CombineOutputs(name='combine_outputs')([span_begin_probabilities, span_end_probabilities])

        model = Model([passage_input, question_input], [output])

        model.summary()

        try:
            model = multi_gpu_model(model)
        except:
            pass

        adadelta = Adadelta(lr=0.01)
        model.compile(loss=negative_avg_log_error, optimizer=adadelta, metrics=[accuracy])

        self.model = model

    def train_model(self, train_generator, steps_per_epoch=None, epochs=1, validation_generator=None, validation_steps=None, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0, save_history=False, save_model_per_epoch=False):

        saved_items_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'saved_items')
        if not os.path.exists(saved_items_dir):
            os.makedirs(saved_items_dir)

        callbacks = []

        if save_history:
            history_file = os.path.join(saved_items_dir, 'history')
            csv_logger = CSVLogger(history_file, append=True)
            callbacks.append(csv_logger)

        if save_model_per_epoch:
            save_model_file = os.path.join(saved_items_dir, 'bidaf_{epoch:02d}.h5')
            checkpointer = ModelCheckpoint(filepath=save_model_file, verbose=1)
            callbacks.append(checkpointer)

        history = self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks, validation_data=validation_generator,
                                           validation_steps=validation_steps, workers=workers, use_multiprocessing=use_multiprocessing, shuffle=shuffle, initial_epoch=initial_epoch)

        if not save_model_per_epoch:
            self.model.save(os.path.join(saved_items_dir, 'bidaf.h5'))

        return history, self.model
