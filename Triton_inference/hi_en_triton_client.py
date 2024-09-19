import numpy as np
from tqdm import tqdm
from tritonclient.utils import *
from random import choice, randrange
import tritonclient.http as httpclient
from multiprocessing.pool import ThreadPool

shape = [1]
MIN_WORDS, MAX_WORDS = 4, 20
model_name = "nmt"

def task():
    results = []
    inp_lines = ['मैं एक नया कौशल सीख रहा हूँ।', 'मैं संगीत सुन रहा हूँ।']
    lang_pair_map = list({'en-hi': 1, 'hi-en': 2, 'te-en': 4, 'hi-te': 6, 'te-hi': 7, 'en-gu': 8, 'gu-en': 9}.keys())
    with httpclient.InferenceServerClient("localhost:8040") as client:
        async_responses = []
        for idx, i in enumerate(inp_lines):
            source_data = np.array([[i]], dtype='object')
            inputs = [
                httpclient.InferInput("INPUT_TEXT", source_data.shape, np_to_triton_dtype(source_data.dtype)),
                httpclient.InferInput("INPUT_LANGUAGE_ID", source_data.shape, np_to_triton_dtype(source_data.dtype)),
                httpclient.InferInput("OUTPUT_LANGUAGE_ID", source_data.shape, np_to_triton_dtype(source_data.dtype))
            ]
            inputs[0].set_data_from_numpy(np.array([[i]], dtype='object'))
            inputs[1].set_data_from_numpy(np.array([['hi']], dtype='object'))
            inputs[2].set_data_from_numpy(np.array([['en']], dtype='object'))

            # Assign a unique request ID for each inference request
            async_responses.append(client.infer(model_name, inputs, request_id=str(idx)))

        cnt = 0
        for r in async_responses:
            print("Sentence in Hindi: ", inp_lines[cnt])
            print("Translation in English: ", r.as_numpy('OUTPUT_TEXT')[0][0].decode("utf-8"))
            print("*****" * 10)
            cnt += 1

task()
