from keras.utils import Sequence
from keras.preprocessing.text import text_to_word_sequence
import os
from pymagnitude import Magnitude, MagnitudeUtils
import numpy as np


class BatchGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, gen_type, batch_size=32):
        'Initialization'
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

        self.context_file = os.path.join(base_dir, 'squad', gen_type + '.context')
        self.question_file = os.path.join(base_dir, 'squad', gen_type + '.question')
        self.span_file = os.path.join(base_dir, 'squad', gen_type + '.span')
        self.batch_size = batch_size
        with open(self.span_file, 'r', encoding='utf-8') as f:
            for i, _ in enumerate(f):
                pass
            self.num_of_batches = (i + 1) // self.batch_size
        glove = Magnitude(MagnitudeUtils.download_model('glove/medium/glove.6B.300d',
                                                        download_dir=os.path.join(base_dir, 'magnitude')), case_insensitive=True)
        fasttext = Magnitude(MagnitudeUtils.download_model('fasttext/medium/wiki-news-300d-1M-subword',
                                                           download_dir=os.path.join(base_dir, 'magnitude')), case_insensitive=True)
        self.vectors = Magnitude(glove, fasttext)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_of_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        contexts = []
        with open(self.context_file, 'r', encoding='utf-8') as cf:
            for i, line in enumerate(cf):
                if i >= start_index:
                    contexts.append(text_to_word_sequence(
                        line, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' '))
                if i == end_index - 1:
                    break

        questions = []
        with open(self.question_file, 'r', encoding='utf-8') as qf:
            for i, line in enumerate(qf):
                if i >= start_index:
                    questions.append(text_to_word_sequence(
                        line, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' '))
                if i == end_index - 1:
                    break

        answer_spans = []
        with open(self.span_file, 'r', encoding='utf-8') as sf:
            for i, line in enumerate(sf):
                if i >= start_index:
                    answer_spans.append(text_to_word_sequence(
                        line, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' '))
                if i == end_index - 1:
                    break

        context_batch = self.vectors.query(contexts)
        question_batch = self.vectors.query(questions)
        span_batch = np.expand_dims(np.array(answer_spans), axis=0)

        return [context_batch, question_batch], span_batch
