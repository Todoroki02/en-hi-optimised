import os
import json
import numpy
from itertools import islice
from ctranslate2 import Translator
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.source_lang, self.target_lang = 'hi', 'en'
        self.model_config = json.loads(args["model_config"])
        self.device_id = int(json.loads(args["model_instance_device_id"]))
        target_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT_SENT"
        )
        self.target_dtype = pb_utils.triton_string_to_numpy(target_config["data_type"])
        try:
            self.translator = Translator(
                f"{os.path.join(current_path, 'translator')}",
                device="cuda",
                intra_threads=1,
                inter_threads=1,
                device_index=[self.device_id],
            )
        except:
            self.translator = Translator(
                f"{os.path.join(current_path, 'translator')}",
                device="cpu",
                intra_threads=4,
            )

    def clean_output(self, text):
        text = text.replace("@@ ", "")
        text = text.replace("\u200c", "")
        text = text.replace(" ?", "?").replace(" !", "!").replace(" .", ".").replace(" ,", ",")
        if text.startswith("<to-gu> "):
            text = text[8:]
        if text.endswith(" <to-gu>"):
            text = text[:-8]
        return text

    def execute(self, requests):
        source_list = [
            pb_utils.get_input_tensor_by_name(request, "INPUT_SENT_TOKENIZED")
            for request in requests
        ]
        bsize_list = [source.as_numpy().shape[0] for source in source_list]
        src_sentences = [
            s[0].decode("utf-8").strip().split(" ")
            for source in source_list
            for s in source.as_numpy()
        ]
        tgt_sentences = [
            self.clean_output(" ".join(result.hypotheses[0]))
            for result in self.translator.translate_iterable(
                src_sentences,
                max_batch_size=128,
                max_input_length=100,
                max_decoding_length=100,
                beam_size=15,
                replace_unknowns=True,
            )
        ]
        responses = [
            pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "OUTPUT_SENT",
                        numpy.array(
                            [[s.encode('utf-8')] for s in islice(tgt_sentences, bsize)], dtype=self.target_dtype
                        ),
                    )
                ]
            )
            for bsize in bsize_list
        ]
        return responses

    def finalize(self):
        self.translator.unload_model()
