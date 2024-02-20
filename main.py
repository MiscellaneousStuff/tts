from ljspeech import LJSPEECH

DATASET_PATH      = "./data/LJSpeech/"
BANDWIDTH_IDX     = 0
BANDWIDTHS        = [1.5, 3.0, 6.0, 12.0, 24.0]
BANDWIDTH         = BANDWIDTHS[BANDWIDTH_IDX]
MAX_PROMPT_LENGTH = 128

if __name__ == "__main__":
    dataset = LJSPEECH("./data/LJSpeech",
                       encodec_bandwidth=BANDWIDTH,
                       max_prompt_length=MAX_PROMPT_LENGTH)