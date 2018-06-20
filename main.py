from source.config.constants import *
from source.data_utils.word_tokens import CreateWordTokens


if __name__ == "__main__":
    gen_tokens = CreateWordTokens(BASE_DIR)
    gen_tokens.load_req_data()
