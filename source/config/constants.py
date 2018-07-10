import os

BASE_DIR = os.getcwd()

DATA_DIR = "{}/data/".format(BASE_DIR)

TOKENS = "tokens"
TOKENFREQ = "tokenfreq"
WORDCOUNT = "wordcount"
REVTOKENS = "revtokens"

# default token name for rare words
UNK = "UNK"

# below are tokens for end of sentence and
# start of sentence
EOS = "</s>"
SOS = "<s>"

# Word2Vec config
# Arbitrary as of now
EMBEDDING_SIZE = 512

# taking only those words which are
# having length greater than 11 and
# np.lower(np.log10(frequency)) > 2
# VOCABULARY_SIZE = 35914  # this is the initial iteration
# below is with frequency = 25
# & log10(frequency) = 1.3979
# and np.log10(frequency) > 1.39
VOCABULARY_SIZE = 386200

MIN_FREQ = 1.39

# based on stats on 2 million addresses,
# if we clip addresses at 40 words, i.e. take
# first 40 words in a address, then total address
# included without clipping would be 99.96%
MAX_ADDR_LENGTH = 40

# print loss after this many iterations
LOSS_ITERATION = 10000

# print word2vec validation after this many iterations
# this causes extra delay, since it compare words
WORD_VALIDATION_ITERATION = 100000

# window size
SKIP_WINDOW_SIZE = 5

# number of negative samples
NUM_NEGATIVE_SAMPLES = 500

# batch_size
BATCH_SIZE = 10000

# Checkpointing directory
CKP_BASE_DIR = "{}/checkpoint/".format(BASE_DIR)

# checkpoints for word2vec models
CKP_WORD2VEC_DIR = "{}word2vec/".format(CKP_BASE_DIR)

# summaries directory
SUMMARY_BASE_DIR = "{}/summary/".format(BASE_DIR)

# summaries encoder decoder
SUMM_ENC_DEC = "{}summ_enc_dec/".format(SUMMARY_BASE_DIR)

# summaries autoencoder
SUMM_AUTO_ENC = "{}summ_auto_enc/".format(SUMMARY_BASE_DIR)

# ALL CHECKPOINT DIRS
CKP_DIRS = [CKP_BASE_DIR,
            CKP_WORD2VEC_DIR,
            SUMMARY_BASE_DIR,
            SUMM_ENC_DEC,
            SUMM_AUTO_ENC
            ]
