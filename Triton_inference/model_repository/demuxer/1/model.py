import json
import numpy
import asyncio
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # Initialize target data type for the output tensor
        self.target_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args["model_config"]), "OUTPUT_TEXT"
            )["data_type"]
        )

    async def execute(self, requests):
        inference_responses = []
        print("bhaskar inside the demuxer")
        # Loop through each request
        for request in requests:
            input_text_tokenized = pb_utils.get_input_tensor_by_name(
                request, "INPUT_TEXT_TOKENIZED"
            ).as_numpy()
            input_language_id = pb_utils.get_input_tensor_by_name(
                request, "INPUT_LANGUAGE_ID"
            ).as_numpy()
            output_language_id = pb_utils.get_input_tensor_by_name(
                request, "OUTPUT_LANGUAGE_ID"
            ).as_numpy()

            inference_requests = []

            # Determine the model to use for each input
            for input_token, in_lang_id, out_lang_id in zip(
                input_text_tokenized, input_language_id, output_language_id
            ):
                print("input lang id is",in_lang_id,"output lang id is ",out_lang_id)
                if in_lang_id[0].decode("utf-8") == "en" and out_lang_id[0].decode("utf-8") == "hi":
                    model_name = "en_hi_nmt"
                    print("en_hi_nmt model loaded from demuxer")
                elif in_lang_id[0].decode("utf-8") == "hi" and out_lang_id[0].decode("utf-8") == "en":
                    model_name = "hi_en_nmt"
                    print("hi_en_nmt model loaded from demuxer")
                else:
                    raise ValueError(
                        f"Unsupported language pair: {in_lang_id} to {out_lang_id}"
                    )

                # Create inference request for the appropriate model
                inference_request = pb_utils.InferenceRequest(
                    model_name=model_name,
                    requested_output_names=["OUTPUT_SENT"],
                    inputs=[
                        pb_utils.Tensor(
                            "INPUT_SENT_TOKENIZED",
                            numpy.array(
                                [[input_token[0]]],
                                dtype=self.target_dtype,
                            ),
                        )
                    ],
                ).async_exec()

                inference_requests.append(inference_request)

            # Perform inference asynchronously
            results = await asyncio.gather(*inference_requests)

            # Prepare the output tensor from the results
            output_text = numpy.array(
                [
                    [
                        pb_utils.get_output_tensor_by_name(
                            result, "OUTPUT_SENT"
                        ).as_numpy()[0, 0]
                    ]
                    for result in results
                ],
                dtype=self.target_dtype,
            )

            # Create and append the inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("OUTPUT_TEXT", output_text)
                ]
            )
            inference_responses.append(inference_response)

        return inference_responses

    def finalize(self):
        pass
