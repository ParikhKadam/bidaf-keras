from ..scripts import BatchGenerator


def load_data_generators(batch_size, emdim, squad_version=1.1, shuffle=False):
    train_generator = BatchGenerator('train', batch_size, emdim, squad_version=squad_version, shuffle=shuffle)
    validation_generator = BatchGenerator('dev', batch_size, emdim, squad_version=squad_version, shuffle=shuffle)
    return train_generator, validation_generator