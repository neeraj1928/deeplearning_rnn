import os
import os.path
import json
from pprint import pprint
import csv
import re
import pickle
import ast

import numpy as np

from source.config.constants import *


class Config():
    data_dir = DATA_DIR
    vocab_size = VOCABULARY_SIZE
    min_freq = MIN_FREQ


class CreateWordTokens():

    def __init__(self):
        self.config = Config()
        self.paths = self.get_files()
        # replace all non alphanumeric with white spaces
        self.reg = re.compile(r'[\W]+')

    def get_files(self):
        item_list = os.listdir(self.config.data_dir)

        file_list = []
        for item in item_list:
            item_dir = os.path.join(self.config.data_dir, item)
            if not os.path.isdir(item_dir):
                file_list.append(item_dir)
        return file_list

    def create_tokens(self):
        """
            create dictionaries containing tokens
            for all word in vocabulary
        """
        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0

        for sentences in self.read_sentences():
            for sentence in sentences:
                for w in sentence:
                    wordcount += 1
                    if w not in tokens:
                        tokens[w] = idx
                        revtokens += [w]
                        tokenfreq[w] = 1
                        idx += 1
                    else:
                        tokenfreq[w] += 1

        tokens[UNK] = idx
        revtokens += [UNK]
        tokenfreq[UNK] = 1
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens

        return self._tokens

    def create_req_vocab_tokens(self):
        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0
        s_token_freq = sorted(
            self._tokenfreq.items(), key=lambda kv: kv[1], reverse=True)
        for word, freq in s_token_freq:
            if np.log10(freq) > self.config.min_freq:
                tokens[word] = idx
                tokenfreq[word] = freq
                wordcount += 1
                revtokens.append(word)
            else:
                break
            idx += 1

        # adding 3 helper tokens
        helper_tokens = [UNK, SOS, EOS]
        for h_token in helper_tokens:
            tokens[h_token] = idx
            revtokens += [h_token]
            tokenfreq[h_token] = 1
            wordcount += 1
            idx += 1

        self.tokens = tokens
        self.tokenfreq = tokenfreq
        self.wordcount = wordcount
        self.revtokens = revtokens

    def save_req_data(self, tokens, tokenfreq,
                      wordcount, revtokens, is_all=False):
        """
            save data required inside base_dir/pickles
            @param is_all: whether full vocabulary is used
                or some part of it
        """
        name_prefix = "_" if is_all else "{}_".format(
            self.config.vocab_size)
        required_obj = {
            TOKENS: tokens,
            TOKENFREQ: tokenfreq,
            WORDCOUNT: wordcount,
            REVTOKENS: revtokens}
        if not os.path.isdir("{}pickles".format(self.config.data_dir)):
            os.makedirs("{}pickles".format(self.config.data_dir))
        for name, obj in required_obj.items():
            pickle.dump(obj, open(
                "{}pickles/{}{}".format(self.config.data_dir,
                                        name_prefix, name), 'wb'))

    def load_req_data(self):
        """
            save data required inside base_dir/pickles
        """
        if not os.path.isdir("{}pickles".format(self.config.data_dir)):
            self.create_tokens()
            self.save_req_data(
                self._tokens, self._tokenfreq,
                self._wordcount, self._revtokens, True)

        self._tokens = pickle.load(
            open("{}/pickles/_{}".format(
                self.config.data_dir, TOKENS), 'rb'))
        self._tokenfreq = pickle.load(
            open("{}/pickles/_{}".format(
                self.config.data_dir, TOKENFREQ), 'rb'))
        self._wordcount = pickle.load(
            open("{}/pickles/_{}".format(
                self.config.data_dir, WORDCOUNT), 'rb'))
        self._revtokens = pickle.load(
            open("{}/pickles/_{}".format(
                self.config.data_dir, REVTOKENS), 'rb'))
        self.create_req_vocab_tokens()
        self.save_req_data(
            self.tokens, self.tokenfreq,
            self.wordcount, self.revtokens, False)

    def read_sentences(self):
        """
            yield sentences in files iteratively
        """
        for path in self.paths:
            data = json.load(open(path, 'r'))
            clean_data = []
            for row in data:
                pin = ast.literal_eval(
                    row["v3_result"])["corrected_pincode_list"]
                pin = pin[0] if pin else row["pin"]
                clean_data.append(self.clean_address(
                    "{} {} {} {}".format(
                        row["address"], row["city"],
                        row["state"], pin)))
            yield clean_data
            print("done: {}".format(path))

    def clean_address(self, sentence):
        """
            this class does some kind of cleaning for all
            words in a sentence. And then returns a list of words

            @param sentence: given address as a string

            @return: list cleaned words in that sentence
        """
        words = [word.lower() for word in
                 # replace all non alphanumeric with white spaces
                 self.reg.subn(" ", sentence)[0].split()]

        return self.cleaning_logic(words)

    def cleaning_logic(self, words):
        """
            # TODO: more logic can be written here
            @param words: list of words in lowercases present in an address
        """
        return words
