# syntax=docker/dockerfile:1
FROM ubuntu:22.04

# install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip libsndfile1
RUN python3 -m pip install torch==1.11.0 torchaudio numpy numba scipy transformers==4.26.1 soundfile yacs
RUN python3 -m pip install pypinyin jieba

# install app
RUN mkdir /EmotiVoice
COPY . /EmotiVoice/

# final configuration
EXPOSE 8501
RUN python3 -m pip install streamlit g2p_en
WORKDIR /EmotiVoice
RUN python3 frontend_en.py
CMD streamlit run demo_page.py --server.port 8501
