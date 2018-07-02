import os

BASE_DIR = os.getcwd()

DATA_DIR = "{}/data/".format(BASE_DIR)

TOKENS = "tokens"
TOKENFREQ = "tokenfreq"
WORDCOUNT = "wordcount"
REVTOKENS = "revtokens"

# default token name for rare words
UNK = "UNK"

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
VOCABULARY_SIZE = 386198

MIN_FREQ = 1.39

# window size
SKIP_WINDOW_SIZE = 2

# number of negative samples
NUM_NEGATIVE_SAMPLES = 500

# batch_size
BATCH_SIZE = 10000

# Checkpointing directory
CKP_BASE_DIR = "{}/checkpoint/".format(BASE_DIR)

# checkpoints for word2vec models
CKP_WORD2VEC_DIR = "{}word2vec/".format(CKP_BASE_DIR)

# ALL CHECKPOINT DIRS
CKP_DIRS = [CKP_BASE_DIR, CKP_WORD2VEC_DIR]
