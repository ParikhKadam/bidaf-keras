from ..scripts import BatchGenerator


def load_data_generators(batch_size, emdim, shuffle=False):
    train_generator = BatchGenerator('train', batch_size, emdim, shuffle)
    validation_generator = BatchGenerator('dev', batch_size, emdim, shuffle)
    return train_generator, validation_generator
