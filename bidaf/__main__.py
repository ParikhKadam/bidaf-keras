from .models import BidirectionalAttentionFlow
from .scripts import load_data_generators
from .scripts import data_download_and_preprocess, negative_avg_log_error, accuracy
import os

# =======================================================================================================================


import argparse
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-sv', '--squad_version', choices=[1.1, 2.0],
                    action='store', default=1.1, help='SQuAD dataset version')
parser.add_argument('-mpl', '--max_passage_length', type=int, action='store',
                    default=None, help='Maximum length of input passage')
parser.add_argument('-mql', '--max_query_length', type=int, action='store',
                    default=None, help='Maximum length of input question')
parser.add_argument('-l', '--do_lowercase', action='store_true', default=False, help='Convert input to lowercase')
parser.add_argument('--model_name', type=str, action='store', default=None,
                    help='Model to load for predictions/resume training')
parser.add_argument('-e', '--emdim', choices=[350, 400, 500, 600],
                    action='store', default=400, help='Embedding (GLoVE + Fasttext) vectors dimension')
parser.add_argument('-nhl', '--num_highway_layers', type=int, action='store',
                    default=1, help='Number of Highway layers')
parser.add_argument('-nd', '--num_decoders', type=int, action='store', default=1, help='Number of decoders')
parser.add_argument('-ed', '--encoder_dropout', type=float, action='store', default=0.0, help='Encoder dropout')
parser.add_argument('-dd', '--decoder_dropout', type=float, action='store', default=0.0, help='Decoder dropout')

subparsers = parser.add_subparsers(help='Specify if you want to train or predict', dest='which')

# create the parser for the "train" command
parser_train = subparsers.add_parser('train', help='Train BiDAF',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_train.add_argument('-bs', '--batch_size', type=int, action='store', default=16, help='Batch size')
parser_train.add_argument('-ss', '--shuffle_samples', action='store_true',
                          default=False, help='Shuffle samples in batch at epoch end')
parser_train.add_argument('-spe', '--steps_per_epochs', type=int, action='store', default=None, help='Steps per epoch')
parser_train.add_argument('-vs', '--validation_steps', type=int, action='store', default=None, help='Validation steps')
parser_train.add_argument('-w', '--workers', type=int, action='store', default=1, help='Number of workers')
parser_train.add_argument('--use_multiprocessing', action='store_true', default=False, help='Use multiprocessing')
parser_train.add_argument('-sb', '--shuffle_batch', action='store_true',
                          default=False, help='Shuffle batches while training')
parser_train.add_argument('-sh', '--save_history', action='store_true', default=False,
                          help='Save history in a csv file while training')
parser_train.add_argument('-smpe', '--save_model_per_epoch', action='store_true',
                          default=False, help='Save checkpoint after every epoch')

required_train = parser_train.add_argument_group('required arguments')
required_train.add_argument('--epochs', type=int, action='store', required=True, help='Total number of epochs')

# create the parser for the "predict" command
parser_predict = subparsers.add_parser('predict', help='Run predictions on BiDAF')
parser_predict.add_argument('-mal', '--max_ans_length', action='store',
                            default=25, help='Maximum answer length')
parser_predict.add_argument('-rcl', '--return_char_loc', action='store_true', default=False,
                            help='Return answer start and end character locations')
parser_predict.add_argument('-rcs', '--return_confidence_score', action='store_true',
                            default=False, help='Return confidence value of the answer')

required_predict = parser_predict.add_argument_group('required arguments')
required_predict.add_argument('-p', '--passage', type=str, action='store', required=True, help='Input passage')
required_predict.add_argument('-q', '--question', type=str, action='store', required=True, help='Input question')


# ========================================================================================================================


def main():
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit()
    args = parser.parse_args()

    data_download_and_preprocess(squad_version=args.squad_version, do_lowercase=args.do_lowercase)

    bidaf_model = BidirectionalAttentionFlow(emdim=args.emdim, max_passage_length=args.max_passage_length,
                                             max_query_length=args.max_query_length,
                                             num_highway_layers=args.num_highway_layers, num_decoders=args.num_decoders,
                                             encoder_dropout=args.encoder_dropout, decoder_dropout=args.decoder_dropout)

    if args.which == 'train':

        if args.model_name is not None:
            bidaf_model.load_bidaf(os.path.join(os.path.dirname(__file__), 'saved_items', args.model_name))
            bidaf_model.model.compile(loss=negative_avg_log_error, optimizer='adadelta', metrics=[accuracy])

        train_generator, validation_generator = load_data_generators(batch_size=args.batch_size, emdim=args.emdim,
                                                                     squad_version=args.squad_version,
                                                                     max_passage_length=args.max_passage_length,
                                                                     max_query_length=args.max_query_length,
                                                                     shuffle=args.shuffle_samples)

        bidaf_model.train_model(train_generator, steps_per_epoch=args.steps_per_epochs, epochs=args.epochs,
                                validation_generator=validation_generator, validation_steps=args.validation_steps,
                                workers=args.workers, use_multiprocessing=args.use_multiprocessing,
                                shuffle=args.shuffle_batch, save_history=args.save_history,
                                save_model_per_epoch=args.save_model_per_epoch)

        print("Training Completed!")

    if args.which == 'predict':
        if args.model_name is None:
            print("You must specify a model to run predictions on it.", file=sys.stderr)
            sys.exit(1)

        print("Your passage:", args.passage)
        print("Your question:", args.question)
        print("Predicting answer...")

        bidaf_model.load_bidaf(os.path.join(os.path.dirname(__file__), 'saved_items', args.model_name))

        answer = bidaf_model.predict_ans(args.passage, args.question, squad_version=args.squad_version,
                                         max_span_length=args.max_ans_length,
                                         do_lowercase=args.do_lowercase, return_char_loc=args.return_char_loc,
                                         return_confidence_score=args.return_confidence_score)

        print("Predicted answer:", answer)


if __name__ == '__main__':
    main()
