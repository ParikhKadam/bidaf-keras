from ..scripts import BatchGenerator


def load_data_generators(batch_size, emdim, squad_version=1.1, max_passage_length=None, max_query_length=None,
                         shuffle=False):
    train_generator = BatchGenerator('train', batch_size, emdim, squad_version, max_passage_length, max_query_length,
                                     shuffle)
    validation_generator = BatchGenerator('dev', batch_size, emdim, squad_version, max_passage_length, max_query_length,
                                          shuffle)
    return train_generator, validation_generator
