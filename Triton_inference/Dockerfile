FROM nvcr.io/nvidia/tritonserver:23.07-py3
WORKDIR /opt/tritonserver
RUN apt-get update 
RUN pip install -U ctranslate2 indic-nlp-library mosestokenizer subword_nmt sentencepiece nltk
#CMD ["tritonserver", "--model-repository=/models", "--http-port=8040", "--grpc-port=8060", "--metrics-port=8080", "--cache-config=local,size=1048576"]
CMD ["tritonserver", "--cache-config=local,size=1048576"]
#EXPOSE 8040
#EXPOSE 8060
#EXPOSE 8080
