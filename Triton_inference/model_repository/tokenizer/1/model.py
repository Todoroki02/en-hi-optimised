import os
import json
import numpy as np
import codecs
import nltk
from subword_nmt.apply_bpe import BPE
from mosestokenizer import MosesSentenceSplitter, MosesTokenizer
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import triton_python_backend_utils as pb_utils

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

class TritonPythonModel:
    def initialize(self, args):
        # Load model configuration
        model_config = json.loads(args["model_config"])
        input_text_tokenized_config = pb_utils.get_output_config_by_name(model_config, "INPUT_TEXT_TOKENIZED")
        self.target_dtype = pb_utils.triton_string_to_numpy(input_text_tokenized_config["data_type"])

        current_path = os.path.dirname(os.path.abspath(__file__))

        # Initialize BPE and normalizer for English-Hindi and Hindi-English
        self.en_hi_bpe_path = f"{current_path}/bpe_src/codes.en"
        self.hi_en_bpe_path = f"{current_path}/bpe_src/codes.hi"

        # Load BPE codes for English-Hindi translation
        self.en_hi_codes = codecs.open(self.en_hi_bpe_path, encoding='utf-8')
        self.en_hi_bpe = BPE(self.en_hi_codes)

        # Load BPE codes for Hindi-English translation
        self.hi_en_codes = codecs.open(self.hi_en_bpe_path, encoding='utf-8')
        self.hi_en_bpe = BPE(self.hi_en_codes)

        # Initialize Indic normalizer for Hindi
        self.factory = IndicNormalizerFactory()
        self.normalizer = self.factory.get_normalizer("hi")

    def preprocess_text(self, text, source_lang, target_lang):
        """Preprocess text for specific source and target language pairs."""
        # Add preprocessing rules if required
        return (
            f"<to-gu> {text} <to-gu>"
            if source_lang == "en" and target_lang == "gu"
            else text
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            # Extract input tensors
            input_texts = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT").as_numpy()
            input_language_ids = pb_utils.get_input_tensor_by_name(request, "INPUT_LANGUAGE_ID").as_numpy()
            output_language_ids = pb_utils.get_input_tensor_by_name(request, "OUTPUT_LANGUAGE_ID").as_numpy()

            tokenized_sents = []

            # Tokenize each input based on language pair
            for input_text, input_language_id, output_language_id in zip(input_texts, input_language_ids, output_language_ids):
                #print("inside the for loop")
                print("ip id:",input_language_id[0].decode("utf-8"),"op id :",output_language_id[0].decode("utf-8"))
                source_lang = input_language_id[0].decode("utf-8")  # Convert to Python string
                print("source lang",source_lang,type(source_lang))
                target_lang = output_language_id[0].decode("utf-8")
                print("target lang",target_lang,type(target_lang))

                text = input_text[0].decode('utf-8').lower()  # Convert to Python string and lowercase

                # Choose tokenizer based on language pair
                if source_lang == "en" and target_lang == "hi":
                    # English to Hindi: Use English tokenizer and BPE
                    print("en_hi_loaded from tokenzier")
                    tokenized_text = ' '.join(nltk.word_tokenize(text))
                    tokenized_sent = self.en_hi_bpe.process_line(tokenized_text).strip()

                elif source_lang == "hi" and target_lang == "en":
                    print("hi_en_loaded from tokenizer")
                    # Hindi to English: Use Indic tokenizer and BPE
                    normalized_text = self.normalizer.normalize(text)
                    tokenized_text = ' '.join(indic_tokenize.trivial_tokenize(normalized_text))
                    tokenized_sent = self.hi_en_bpe.process_line(tokenized_text).strip()

                else:
                    print("else executed")
                    raise ValueError(f"Unsupported language pair: {source_lang} to {target_lang}")

                tokenized_sents.append([tokenized_sent])

            # Create inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "INPUT_TEXT_TOKENIZED",
                        np.array(tokenized_sents, dtype=self.target_dtype)
                    )
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass
