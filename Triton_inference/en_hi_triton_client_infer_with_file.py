import numpy as np
from tqdm import tqdm
from tritonclient.utils import *
from random import choice
import tritonclient.http as httpclient

# File paths
input_file_path = '/home/tto/jank/nmt/floresp-v2.0-rc.3/dev/dev.eng_Latn'
output_file_path = 'en_hi_dev_output.txt'

# Parameters
MIN_WORDS, MAX_WORDS = 4, 20
model_name = "nmt"

def task():
    results = []

    # Read input lines from the file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        inp_lines = [line.strip() for line in f if line.strip()]

    lang_pair_map = list({'en-hi': 1, 'hi-en': 2, 'te-en': 4, 'hi-te': 6, 'te-hi': 7, 'en-gu': 8, 'gu-en': 9}.keys())

    with httpclient.InferenceServerClient("localhost:8040") as client:
        async_responses = []
        for i in inp_lines:
            source_data = np.array([[i]], dtype='object')
            inputs = [
                httpclient.InferInput("INPUT_TEXT", source_data.shape, np_to_triton_dtype(source_data.dtype)),
                httpclient.InferInput("INPUT_LANGUAGE_ID", source_data.shape, np_to_triton_dtype(source_data.dtype)),
                httpclient.InferInput("OUTPUT_LANGUAGE_ID", source_data.shape, np_to_triton_dtype(source_data.dtype))
            ]
            inputs[0].set_data_from_numpy(np.array([[i]], dtype='object'))
            inputs[1].set_data_from_numpy(np.array([['en']], dtype='object'))
            inputs[2].set_data_from_numpy(np.array([['hi']], dtype='object'))

            async_responses.append(client.infer(model_name, inputs, request_id=str(1)))

        # Open the output file for writing
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for cnt, r in enumerate(async_responses):
                translated_text = r.as_numpy('OUTPUT_TEXT')[0][0].decode('utf-8').replace("@@ ", "")
                #output_file.write(f"Original Sentence: {inp_lines[cnt]}\n")
                output_file.write(translated_text+"\n")
                #output_file.write("*****" * 10 + "\n")
                #print(f"Original Sentence: {inp_lines[cnt]}")
                #print(f"Translation: {translated_text}")

task()
