from ..scripts import BatchGenerator


def load_data_generators(batch_size, emdim):
    train_generator = BatchGenerator('train', batch_size, emdim)
    validation_generator = BatchGenerator('dev', batch_size, emdim)
    return train_generator, validation_generator
