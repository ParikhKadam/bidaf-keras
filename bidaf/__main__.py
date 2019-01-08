from .models import BidirectionalAttentionFlow


def main():
    bidaf = BidirectionalAttentionFlow(emdim=600, num_highway_layers=2, num_decoders=1)
    bidaf.load_data_generators(batch_size=40)
    model = bidaf.train_model(epochs=40, save_history=True, save_model_per_epoch=True)


if __name__ == '__main__':
    main()
