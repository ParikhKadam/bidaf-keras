import os
from pymagnitude import Magnitude, MagnitudeUtils


class MagnitudeVectors():

    def __init__(self, emdim):

        base_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')

        self.fasttext_dim = 300
        self.glove_dim = emdim - 300

        assert self.glove_dim in [50, 100, 200,
                                  300], "Embedding dimension must be one of the following: 350, 400, 500, 600"

        print("Will download magnitude files from the server if they aren't avaialble locally.. So, grab a cup of coffee while the downloading is under progress..")
        glove = Magnitude(MagnitudeUtils.download_model('glove/medium/glove.6B.{}d'.format(self.glove_dim),
                                                        download_dir=os.path.join(base_dir, 'magnitude')), case_insensitive=True)
        fasttext = Magnitude(MagnitudeUtils.download_model('fasttext/medium/wiki-news-300d-1M-subword',
                                                           download_dir=os.path.join(base_dir, 'magnitude')), case_insensitive=True)
        self.vectors = Magnitude(glove, fasttext)

    def load_vectors(self):
        return self.vectors
