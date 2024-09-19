import numpy as np
from tritonclient.utils import *
import tritonclient.http as httpclient
from random import choice

# Define the file paths
input_file_path = '/home/tto/jank/nmt/floresp-v2.0-rc.3/dev/dev.hin_Deva'  # Input Hindi file path
output_file_path = 'hi_en_translation_output.txt'  # Output file path

model_name = "nmt"

def task(input_file_path, output_file_path):
    # Read input sentences from the text file
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        inp_lines = infile.readlines()

    lang_pair_map = list({'en-hi': 1, 'hi-en': 2, 'te-en': 4, 'hi-te': 6, 'te-hi': 7, 'en-gu': 8, 'gu-en': 9}.keys())
    results = []

    with httpclient.InferenceServerClient("localhost:8040") as client:
        async_responses = []

        # Process each line in the input file
        for line in inp_lines:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:  # Skip empty lines
                continue

            source_data = np.array([[line]], dtype='object')
            inputs = [
                httpclient.InferInput("INPUT_TEXT", source_data.shape, np_to_triton_dtype(source_data.dtype)),
                httpclient.InferInput("INPUT_LANGUAGE_ID", source_data.shape, np_to_triton_dtype(source_data.dtype)),
                httpclient.InferInput("OUTPUT_LANGUAGE_ID", source_data.shape, np_to_triton_dtype(source_data.dtype))
            ]

            inputs[0].set_data_from_numpy(np.array([[line]], dtype='object'))
            langpair = choice(lang_pair_map)
            inputs[1].set_data_from_numpy(np.array([['hi']], dtype='object'))
            inputs[2].set_data_from_numpy(np.array([['en']], dtype='object'))

            async_responses.append(client.infer(model_name, inputs, request_id=str(1)))

        # Write translated output to the file
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for i, r in enumerate(async_responses):
                translation = r.as_numpy('OUTPUT_TEXT')[0][0].decode('utf-8')
                #original_sentence = inp_lines[i].strip()
                #outfile.write(f"Original (Hindi): {original_sentence}\n")
                outfile.write(translation+"\n")
                #outfile.write("*****" * 10 + "\n")
                #print(f"Translated sentence {i + 1} stored.")

task(input_file_path, output_file_path)
