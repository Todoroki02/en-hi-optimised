import numpy as np
#import wonderwords
from tqdm import tqdm
from tritonclient.utils import *
from random import choice, randrange
import tritonclient.http as httpclient
from multiprocessing.pool import ThreadPool

shape = [1]
MIN_WORDS, MAX_WORDS = 4, 20
model_name = "nmt"
#rs = wonderwords.RandomWord()

def task():
    results = []
    inp_lines = ['I am living in hyderabad', 'lets go outside']
    lang_pair_map = list({'en-hi': 1, 'hi-en': 2, 'te-en': 4, 'hi-te': 6, 'te-hi': 7, 'en-gu': 8, 'gu-en': 9}.keys())
    with httpclient.InferenceServerClient("localhost:8040") as client:
        async_responses = []
        for i in inp_lines:
            #s = ' '.join(rs.random_words(randrange(MIN_WORDS, MAX_WORDS)) + ['.']) # 'this is a sentence.' Use a constant sentence if you want to hit the cache
            #print("s: ",s)
            source_data = np.array([[i]], dtype='object')
            print(i)
            print(source_data)
            print(source_data.shape)
            #print("Source_data",source_data)
            inputs = [httpclient.InferInput("INPUT_TEXT", source_data.shape, np_to_triton_dtype(source_data.dtype)), httpclient.InferInput("INPUT_LANGUAGE_ID", source_data.shape, np_to_triton_dtype(source_data.dtype)), httpclient.InferInput("OUTPUT_LANGUAGE_ID", source_data.shape, np_to_triton_dtype(source_data.dtype))]
            inputs[0].set_data_from_numpy(np.array([[i]], dtype='object'))
            langpair = choice(lang_pair_map)
            #print("Normal: ",langpair.split('-')[0].strip())
            #print("input sentence language: ",np.array([['hi']]))
            #print("output sentence language: ",np.array([['en']]))
            inputs[1].set_data_from_numpy(np.array([['en']], dtype='object'))
            inputs[2].set_data_from_numpy(np.array([['hi']], dtype='object'))
            print(inputs)
            #outputs = [httpclient.InferRequestedOutput("OUTPUT_TEXT")]
            #async_responses.append(client.async_infer(model_name, inputs, request_id=str(1), outputs=outputs))
            async_responses.append(client.infer(model_name, inputs, request_id=str(1)))
        cnt = 0
        for r in async_responses:
            #print(r.get_result(timeout=10).get_response()['outputs'][0])
            print("Sentence in Hindi: ",inp_lines[cnt])
            print("Translation in english: ",r.as_numpy('OUTPUT_TEXT')[0][0].decode('utf-8'))
            print("Translation in Hindi is :",r.as_numpy('OUTPUT_TEXT')[0][0].decode('utf-8').replace("@@ ", ""))
            print("*****"*10)
            cnt+=1

task()
