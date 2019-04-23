from keras.layers import Input, TimeDistributed, LSTM, Bidirectional
from keras.models import Model, load_model
from keras.optimizers import Adadelta
from keras.callbacks import CSVLogger, ModelCheckpoint
from ..layers import Highway, Similarity, C2QAttention, Q2CAttention, MergedContext, SpanBegin, SpanEnd, CombineOutputs
from ..scripts import negative_avg_log_error, accuracy, tokenize, MagnitudeVectors, get_best_span, \
    get_word_char_loc_mapping
from ..scripts import ModelMGPU
import os


class BidirectionalAttentionFlow():

    def __init__(self, emdim, max_passage_length=None, max_query_length=None, num_highway_layers=2, num_decoders=1,
                 encoder_dropout=0, decoder_dropout=0):
        self.emdim = emdim
        self.max_passage_length = max_passage_length
        self.max_query_length = max_query_length

        passage_input = Input(shape=(self.max_passage_length, emdim), dtype='float32', name="passage_input")
        question_input = Input(shape=(self.max_query_length, emdim), dtype='float32', name="question_input")

        question_embedding = question_input
        passage_embedding = passage_input
        for i in range(num_highway_layers):
            highway_layer = Highway(name='highway_{}'.format(i))
            question_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_qtd")
            question_embedding = question_layer(question_embedding)
            passage_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_ptd")
            passage_embedding = passage_layer(passage_embedding)

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
            model = ModelMGPU(model)
        except:
            pass

        adadelta = Adadelta(lr=0.01)
        model.compile(loss=negative_avg_log_error, optimizer=adadelta, metrics=[accuracy])

        self.model = model

    def load_bidaf(self, path):
        custom_objects = {
            'Highway': Highway,
            'Similarity': Similarity,
            'C2QAttention': C2QAttention,
            'Q2CAttention': Q2CAttention,
            'MergedContext': MergedContext,
            'SpanBegin': SpanBegin,
            'SpanEnd': SpanEnd,
            'CombineOutputs': CombineOutputs,
            'negative_avg_log_error': negative_avg_log_error,
            'accuracy': accuracy
        }

        self.model = load_model(path, custom_objects=custom_objects)

    def train_model(self, train_generator, steps_per_epoch=None, epochs=1, validation_generator=None,
                    validation_steps=None, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0,
                    save_history=False, save_model_per_epoch=False):

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

        history = self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                           callbacks=callbacks, validation_data=validation_generator,
                                           validation_steps=validation_steps, workers=workers,
                                           use_multiprocessing=use_multiprocessing, shuffle=shuffle,
                                           initial_epoch=initial_epoch)
        if not save_model_per_epoch:
            self.model.save(os.path.join(saved_items_dir, 'bidaf.h5'))

        return history, self.model

    def predict_ans(self, passage, question, squad_version=1.1, max_span_length=25, do_lowercase=True,
                    return_char_loc=False, return_confidence_score=False):

        if type(passage) == list:
            assert all(type(pas) == str for pas in passage), "Input 'passage' must be of type 'string'"

            passage = [pas.strip() for pas in passage]
            contexts = []
            for pas in passage:
                context_tokens = tokenize(pas, do_lowercase)
                contexts.append(context_tokens)

            if do_lowercase:
                original_passage = [pas.lower() for pas in passage]
            else:
                original_passage = passage

        elif type(passage) == str:
            passage = passage.strip()
            context_tokens = tokenize(passage, do_lowercase)
            contexts = [context_tokens, ]

            if do_lowercase:
                original_passage = [passage.lower(), ]
            else:
                original_passage = [passage, ]

        else:
            raise TypeError("Input 'passage' must be either a 'string' or 'list of strings'")

        assert type(passage) == type(
            question), "Both 'passage' and 'question' must be either 'string' or a 'list of strings'"

        if type(question) == list:
            assert all(type(ques) == str for ques in question), "Input 'question' must be of type 'string'"
            assert len(passage) == len(
                question), "Both lists (passage and question) must contain same number of elements"

            questions = []
            for ques in question:
                question_tokens = tokenize(ques, do_lowercase)
                questions.append(question_tokens)

        elif type(question) == str:
            question_tokens = tokenize(question, do_lowercase)
            questions = [question_tokens, ]

        else:
            raise TypeError("Input 'question' must be either a 'string' or 'list of strings'")

        vectors = MagnitudeVectors(self.emdim).load_vectors()
        context_batch = vectors.query(contexts, self.max_passage_length)
        question_batch = vectors.query(questions, self.max_query_length)

        y = self.model.predict([context_batch, question_batch])
        y_pred_start = y[:, 0, :]
        y_pred_end = y[:, 1, :]

        # clearing the session releases memory by removing the model from memory
        # using this, you will need to load model every time before prediction
        # K.clear_session()

        batch_answer_span = []
        batch_confidence_score = []
        for sample_id in range(len(contexts)):
            answer_span, confidence_score = get_best_span(y_pred_start[sample_id, :], y_pred_end[sample_id, :],
                                                          len(contexts[sample_id]), squad_version, max_span_length)
            batch_answer_span.append(answer_span)
            batch_confidence_score.append(confidence_score)

        answers = []
        for index, answer_span in enumerate(batch_answer_span):
            context_tokens = contexts[index]
            start, end = answer_span[0], answer_span[1]

            # word index to character index mapping
            mapping = get_word_char_loc_mapping(original_passage[index], context_tokens)

            char_loc_start = mapping[start]
            # [1] => char_loc_end is set to point to one more character after the answer
            char_loc_end = mapping[end] + len(context_tokens[end])
            # [1] will help us getting a perfect slice without unnecessary increments/decrements
            ans = original_passage[index][char_loc_start:char_loc_end]

            return_dict = {
                "answer": ans,
            }

            if return_char_loc:
                return_dict["char_loc_start"] = char_loc_start
                return_dict["char_loc_end"] = char_loc_end - 1

            if return_confidence_score:
                return_dict["confidence_score"] = batch_confidence_score[index]

            answers.append(return_dict)

        if type(passage) == list:
            return answers
        else:
            return answers[0]
