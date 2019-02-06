from .models import BidirectionalAttentionFlow
from .scripts.data_generator import load_data_generators


def main():
    emdim = 400
    bidaf = BidirectionalAttentionFlow(emdim=emdim, num_highway_layers=2,
                                       num_decoders=1, encoder_dropout=0.4, decoder_dropout=0.6)
    # train_generator, validation_generator = load_data_generators(batch_size=1, emdim=emdim, shuffle=True)
    # model = bidaf.train_model(train_generator, steps_per_epoch=1, epochs=2, validation_generator=validation_generator,
    #                           save_history=True, save_model_per_epoch=True)
    bidaf.model.load_weights('bidaf_01.h5')


if __name__ == '__main__':
    main()
