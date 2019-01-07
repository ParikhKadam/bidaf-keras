from keras.layers import Input, TimeDistributed, LSTM, Bidirectional
from keras.models import Model
from ..layers import Highway, Similarity, C2QAttention, Q2CAttention, MergedContext, SpanBegin, SpanEnd, CombineOutputs
from ..scripts import negative_avg_log_error, BatchGenerator
import pickle
import os


class BidirectionalAttentionFlow():

    def __init__(self, emdim=600, num_highway_layers=2, num_decoders=1):
        self.emdim = emdim

        question_input = Input(shape=(None, emdim), dtype='float32', name="question_input")
        passage_input = Input(shape=(None, emdim), dtype='float32', name="passage_input")

        for i in range(num_highway_layers):
            highway_layer = Highway(name='highway_{}'.format(i))
            question_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_qtd")
            question_embedding = question_layer(question_input)
            passage_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_ptd")
            passage_embedding = passage_layer(passage_input)

        encoder_layer = Bidirectional(LSTM(emdim, return_sequences=True), name='bidirectional_encoder')
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
            hidden_layer = Bidirectional(LSTM(emdim, return_sequences=True), name='bidirectional_decoder_{}'.format(i))
            modeled_passage = hidden_layer(modeled_passage)

        span_begin_probabilities = SpanBegin(name='span_begin')([merged_context, modeled_passage])
        span_end_probabilities = SpanEnd(name='span_end')(
            [encoded_passage, merged_context, modeled_passage, span_begin_probabilities])

        output = CombineOutputs(name='combine_outputs')([span_begin_probabilities, span_end_probabilities])

        model = Model([passage_input, question_input], [output])

        model.summary()

        model.compile(loss=negative_avg_log_error, optimizer='adadelta', metrics=[])

        self.model = model

    def load_data_generators(self, batch_size=32):
        self.train_generator = BatchGenerator('train', batch_size)
        self.validate_generator = BatchGenerator('dev', batch_size)

    def train_model(self, epochs, steps_per_epoch=None, validation_steps=None, save_history=False, save_model_per_epoch=False):

        saved_items_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'saved_items')
        if not os.path.exists(saved_items_dir):
            os.makedirs(saved_items_dir)

        if save_model_per_epoch:
            history = self.model.fit_generator(self.train_generator, steps_per_epoch=steps_per_epoch, epochs=1,
                                               validation_data=self.validate_generator, validation_steps=validation_steps, shuffle=False)

            if save_history:
                with open(os.path.join(saved_items_dir, 'history'), 'wb') as f:
                    pickle.dump(history.history, f)
            self.model.save(os.path.join(saved_items_dir, 'bidaf_0.h5'))

            for i in range(1, epochs):
                if save_history:
                    with open(os.path.join(saved_items_dir, 'history'), 'rb') as f:
                        old_history = pickle.load(f)

                history = self.model.fit_generator(self.train_generator, steps_per_epoch=steps_per_epoch, epochs=1,
                                                   validation_data=self.validate_generator, validation_steps=validation_steps, shuffle=False)
                old_history['loss'].extend(history.history['loss'])
                old_history['val_loss'].extend(history.history['val_loss'])

                if save_history:
                    with open(os.path.join(saved_items_dir, 'history'), 'wb') as f:
                        pickle.dump(old_history, f)

                self.model.save(os.path.join(saved_items_dir, 'bidaf_{}.h5'.format(i)))

                return old_history, self.model

            else:
                history = self.model.fit_generator(self.train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                                   validation_data=self.validate_generator, validation_steps=validation_steps, shuffle=False)
                self.model.save(os.path.join(saved_items_dir, 'bidaf.h5'))

                return history, self.model
