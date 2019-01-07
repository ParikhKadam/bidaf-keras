from keras.utils import Sequence
import os
from pymagnitude import Magnitude, MagnitudeUtils
import numpy as np


class BatchGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, gen_type, batch_size=32, emdim=600):
        'Initialization'

        self.fasttext_dim = 300
        self.glove_dim = emdim - 300

        assert self.glove_dim in [50, 100, 200,
                                  300], "Embedding dimension must be one of the following: 350, 400, 500, 600"

        base_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')

        self.context_file = os.path.join(base_dir, 'squad', gen_type + '.context')
        self.question_file = os.path.join(base_dir, 'squad', gen_type + '.question')
        self.span_file = os.path.join(base_dir, 'squad', gen_type + '.span')
        self.batch_size = batch_size
        i = 0
        with open(self.span_file, 'r', encoding='utf-8') as f:

            for i, _ in enumerate(f):
                pass
        self.num_of_batches = (i + 1) // self.batch_size

        print("Will download magnitude files from the server if they aren't avaialble locally.. So, wait for a while. The downloading might be under progress..")
        glove = Magnitude(MagnitudeUtils.download_model('glove/medium/glove.6B.{}d'.format(self.glove_dim),
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
        start_index = (index * self.batch_size) + 1
        end_index = ((index + 1) * self.batch_size) + 1

        contexts = []
        with open(self.context_file, 'r', encoding='utf-8') as cf:
            for i, line in enumerate(cf, start=1):
                line = line[:-1]
                if i >= start_index:
                    contexts.append(line.split(' '))
                if i == end_index - 1:
                    break

        questions = []
        with open(self.question_file, 'r', encoding='utf-8') as qf:
            for i, line in enumerate(qf, start=1):
                line = line[:-1]
                if i >= start_index:
                    questions.append(line.split(' '))
                if i == end_index - 1:
                    break

        answer_spans = []
        with open(self.span_file, 'r', encoding='utf-8') as sf:
            for i, line in enumerate(sf, start=1):
                line = line[:-1]
                if i >= start_index:
                    answer_spans.append(line.split(' '))
                if i == end_index - 1:
                    break

        context_batch = self.vectors.query(contexts)
        question_batch = self.vectors.query(questions)
        span_batch = np.expand_dims(np.array(answer_spans, dtype='float32'), axis=1)
        return [context_batch, question_batch], [span_batch]
