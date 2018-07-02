import os

from source.config.constants import *
from source.data_utils.word_tokens import CreateWordTokens
from source.training.word2vec import Word2Vec


if __name__ == "__main__":
    for ckp_dir in CKP_DIRS:
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)
    gen_tokens = CreateWordTokens()
    gen_tokens.load_req_data()
    word2vec = Word2Vec(gen_tokens)
    word2vec.run_model()
