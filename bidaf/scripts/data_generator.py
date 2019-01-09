from ..scripts import BatchGenerator


def load_data_generators(self, batch_size=32):
    train_generator = BatchGenerator('train', batch_size)
    validation_generator = BatchGenerator('dev', batch_size)
    return train_generator, validation_generator
