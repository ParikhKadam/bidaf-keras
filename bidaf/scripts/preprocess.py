"""Downloads SQuAD train and dev sets, preprocesses and writes tokenized versions to file"""

import os
import random
import json
import nltk
import numpy as np
from tqdm import tqdm
from six.moves.urllib.request import urlretrieve

random.seed(42)
np.random.seed(42)
nltk.download('punkt')

SQUAD_BASE_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

base_dir = os.path.join(os.path.dirname(__file__), os.pardir)


def write_to_file(out_file, line):
    """Take a line and file as input, encdes the line to utf-8 and then writes that line to the file"""
    out_file.write(line + '\n')


def data_from_json(filename):
    """Loads JSON data from filename and returns"""
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


def tokenize(sequence, do_lowercase):
    """Tokenizes the input sequence using nltk's word_tokenize function, replaces two single quotes with a double quote"""

    if do_lowercase:
        tokens = [token.replace("``", '"').replace("''", '"').lower()
                  for token in nltk.word_tokenize(sequence)]
    else:
        tokens = [token.replace("``", '"').replace("''", '"')
                  for token in nltk.word_tokenize(sequence)]
    return tokens


def total_examples(dataset):
    """Returns the total number of (context, question, answer) triples, given the data loaded from the SQuAD json file"""
    total = 0
    for article in dataset['data']:
        for para in article['paragraphs']:
            total += len(para['qas'])
    return total


def maybe_download(base_url, filename, destination_dir, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            """
            b: int, optional
                Number of blocks just transferred [default: 1].
            bsize: int, optional
                Size of each block (in tqdm units) [default: 1].
            tsize: int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
            """
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    local_filename = None
    if not os.path.exists(os.path.join(destination_dir, filename)):
        try:
            if show_progress:
                print("Downloading file {}...".format(base_url + filename))
                # Download with a progress bar
                with DownloadProgressBar(unit='B', unit_scale=True,
                                         miniters=1, desc=filename) as t:
                    local_filename, _ = urlretrieve(base_url + filename,
                                                    filename=os.path.join(destination_dir, filename),
                                                    reporthook=t.update_to)
            else:
                # Simple download with no progress bar
                local_filename, _ = urlretrieve(base_url + filename, filename=os.path.join(destination_dir, filename))

            print("File {} successfully loaded".format(filename))
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e
        except KeyboardInterrupt as k:
            if os.path.exists(os.path.join(destination_dir, filename)):
                os.remove(os.path.join(destination_dir, filename))
            raise k


def get_char_word_loc_mapping(context, context_tokens):
    """
    Return a mapping that maps from character locations to the corresponding token locations.
    If we're unable to complete the mapping e.g. because of special characters, we return None.

    Inputs:
      context: string (unicode)
      context_tokens: list of strings (unicode)

    Returns:
      mapping: dictionary from ints (character locations) to (token, token_idx) pairs
        Only ints corresponding to non-space character locations are in the keys
        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
        0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
    """
    acc = ''  # accumulator
    current_token_idx = 0  # current word loc
    mapping = dict()

    # step through original characters
    for char_idx, char in enumerate(context):
        if char != u' ' and char != u'\n':  # if it's not a space:
            acc += char  # add to accumulator
            context_token = context_tokens[current_token_idx]  # current word token
            if acc == context_token:  # if the accumulator now matches the current word token
                # char loc of the start of this word
                syn_start = char_idx - len(acc) + 1
                for char_loc in range(syn_start, char_idx + 1):
                    mapping[char_loc] = (acc, current_token_idx)  # add to mapping
                acc = ''  # reset accumulator
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping


def preprocess_and_write(dataset, tier, out_dir, squad_version, do_lowercase):
    """Reads the dataset, extracts context, question, answer, tokenizes them, and calculates answer span in terms of token indices.
    Note: due to tokenization issues, and the fact that the original answer spans are given in terms of characters, some examples are discarded because we cannot get a clean span in terms of tokens.

    This function produces the {train/dev}.{context/question/answer/span} files.

    Inputs:
      dataset: read from JSON
      tier: string ("train" or "dev")
      out_dir: directory to write the preprocessed files
    Returns:
      the number of (context, question, answer) triples written to file by the dataset.
    """

    num_exs = 0  # number of examples written to file
    num_mappingprob, num_tokenprob, num_spanalignprob = 0, 0, 0
    examples = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):

        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):

            context = article_paragraphs[pid]['context'].strip()  # string

            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context, do_lowercase=do_lowercase)  # list of strings (lowercase)

            if do_lowercase:
                context = context.lower()

            qas = article_paragraphs[pid]['qas']  # list of questions

            # charloc2wordloc maps the character location (int) of a context token to a pair giving (word (string), word loc (int)) of that token
            charloc2wordloc = get_char_word_loc_mapping(
                context, context_tokens)

            if charloc2wordloc is None:  # there was a problem
                num_mappingprob += len(qas)
                continue  # skip this context example

            # for each question, process the question and answer and write to file
            for qn in qas:

                # read the question text and tokenize
                question = qn['question'].strip()  # string
                question_tokens = tokenize(question, do_lowercase=do_lowercase)  # list of strings

                # of the three answers, just take the first
                # get the answer text
                # answer start loc (character count)
                if squad_version == 1.1:
                    ans_text = qn['answers'][0]['text']
                    ans_start_charloc = qn['answers'][0]['answer_start']

                elif qn['is_impossible'] == True:
                    # some questions in squad 2.0 don't even have plausible answers
                    if qn['plausible_answers'] == []:
                        continue

                    is_impossible = 1
                    ans_text = qn['plausible_answers'][0]['text']
                    ans_start_charloc = qn['plausible_answers'][0]['answer_start']
                else:
                    is_impossible = 0
                    ans_text = qn['answers'][0]['text']
                    ans_start_charloc = qn['answers'][0]['answer_start']

                if do_lowercase:
                    ans_text = ans_text.lower()

                # answer end loc (character count) (exclusive)
                ans_end_charloc = ans_start_charloc + len(ans_text)

                # Check that the provided character spans match the provided answer text
                if context[ans_start_charloc:ans_end_charloc] != ans_text:
                    # Sometimes this is misaligned, mostly because "narrow builds" of Python 2 interpret certain Unicode characters to have length 2 https://stackoverflow.com/questions/29109944/python-returns-length-of-2-for-single-unicode-character-string
                    # We should upgrade to Python 3 next year!
                    num_spanalignprob += 1
                    continue

                # get word locs for answer start and end (inclusive)
                # answer start word loc
                ans_start_wordloc = charloc2wordloc[ans_start_charloc][1]
                # answer end word loc
                ans_end_wordloc = charloc2wordloc[ans_end_charloc - 1][1]
                assert ans_start_wordloc <= ans_end_wordloc

                # Check retrieved answer tokens match the provided answer text.
                # Sometimes they won't match, e.g. if the context contains the phrase "fifth-generation"
                # and the answer character span is around "generation",
                # but the tokenizer regards "fifth-generation" as a single token.
                # Then ans_tokens has "fifth-generation" but the ans_text is "generation", which doesn't match.
                ans_tokens = context_tokens[ans_start_wordloc:ans_end_wordloc + 1]
                if "".join(ans_tokens) != "".join(ans_text.split()):
                    num_tokenprob += 1
                    continue  # skip this question/answer pair

                if squad_version == 2.0:
                    examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(
                        ans_tokens), ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)]), str(is_impossible)))
                else:
                    examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(
                        ans_tokens), ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)])))

                num_exs += 1

    print("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", num_mappingprob)
    print("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_tokenprob)
    print("Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): ", num_spanalignprob)
    print("Processed %i examples of total %i\n" %
          (num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalignprob))

    # shuffle examples
    indices = list(range(len(examples)))
    np.random.shuffle(indices)

    with open(os.path.join(out_dir, tier + '-v{}.context'.format(squad_version)), 'w', encoding='utf-8') as context_file, \
            open(os.path.join(out_dir, tier + '-v{}.question'.format(squad_version)), 'w', encoding='utf-8') as question_file, \
            open(os.path.join(out_dir, tier + '-v{}.answer'.format(squad_version)), 'w', encoding='utf-8') as ans_text_file, \
            open(os.path.join(out_dir, tier + '-v{}.span'.format(squad_version)), 'w', encoding='utf-8') as span_file:

        if squad_version == 2.0:
            is_impossible_file = open(os.path.join(
                out_dir, tier + '-v{}.is_impossible'.format(squad_version)), 'w', encoding='utf-8')

        for i in indices:

            if squad_version == 2.0:
                (context, question, answer, answer_span, is_impossible) = examples[i]
            else:
                (context, question, answer, answer_span) = examples[i]

            # write tokenized data to file
            write_to_file(context_file, context)
            write_to_file(question_file, question)
            write_to_file(ans_text_file, answer)
            write_to_file(span_file, answer_span)

            if squad_version == 2.0:
                write_to_file(is_impossible_file, is_impossible)

        if squad_version == 2.0:
            is_impossible_file.close()


def data_download_and_preprocess(squad_version=1.1, do_lowercase=True):
    data_dir = os.path.join(base_dir, 'data', 'squad')

    print("Will download SQuAD datasets to {} if required".format(data_dir))
    print("Will put preprocessed SQuAD datasets in {}".format(data_dir))

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_filename = "train-v{}.json".format(squad_version)
    dev_filename = "dev-v{}.json".format(squad_version)

    # download train set
    maybe_download(SQUAD_BASE_URL, train_filename, data_dir)

    # read train set
    train_data = data_from_json(os.path.join(data_dir, train_filename))
    print("Train data has %i examples total" % total_examples(train_data))

    # preprocess train set and write to file
    if not os.path.isfile(os.path.join(data_dir, 'train-v{}.context'.format(squad_version))):
        print("Preprocessing training data")
        preprocess_and_write(train_data, 'train', data_dir, squad_version, do_lowercase=do_lowercase)
    print("Train data preprocessed!")

    # download dev set
    maybe_download(SQUAD_BASE_URL, dev_filename, data_dir)

    # read dev set
    dev_data = data_from_json(os.path.join(data_dir, dev_filename))
    print("Dev data has %i examples total" % total_examples(dev_data))

    # preprocess dev set and write to file
    if not os.path.isfile(os.path.join(data_dir, 'dev-v{}.context'.format(squad_version))):
        print("Preprocessing development data")
        preprocess_and_write(dev_data, 'dev', data_dir, squad_version, do_lowercase=do_lowercase)
    print("Dev data preprocessed!")
