# Copyright 2023, YOUDAO
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import os, glob
import numpy as np
from yacs import config as CONFIG
import torch
import re

from frontend import g2p
from frontend_en import preprocess_english
from config.joint.config import Config
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from transformers import AutoTokenizer

import base64
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WAV_VALUE = 32768.0

config = Config()

def create_download_link():
    pdf_path = Path("EmotiVoice_UserAgreement_易魔声用户协议.pdf")
    base64_pdf = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="EmotiVoice_UserAgreement_易魔声用户协议.pdf.pdf">EmotiVoice_UserAgreement_易魔声用户协议.pdf</a>'

html=create_download_link()

st.set_page_config(
    page_title="demo page",
    page_icon="📕",
)
st.write("# Text-To-Speech")
st.markdown(f"""
### How to use:
         
- Simply select a speaker, type in the text you want to convert and the emotion prompt, like a single word or even a sentence.
         
- Then click on the synthesize button below to start voice synthesis.
         
- You can download the audio by clicking on the vertical three points next to the displayed audio widget.
         
- This interactive demo page is provided under the {html} file. The audio is synthesized by AI. 音频由AI合成，仅供参考。
         
""", unsafe_allow_html=True)

def g2p_cn(text):
    return g2p(text)

def g2p_en(text):
    return preprocess_english(text)

def scan_checkpoint(cp_dir, prefix, c=8):
    pattern = os.path.join(cp_dir, prefix + '?'*c)
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = re.search(pattern, text)
    return match is not None

@st.cache_resource
def get_models():
    
    am_checkpoint_path = scan_checkpoint(f'{config.output_directory}/prompt_tts_open_source_joint/ckpt', 'g_')

    style_encoder_checkpoint_path = scan_checkpoint(f'{config.output_directory}/style_encoder/ckpt', 'checkpoint_', 6)#f'{config.output_directory}/style_encoder/ckpt/checkpoint_163431' 

    with open(config.model_config_path, 'r') as fin:
        conf = CONFIG.load_cfg(fin)
    
    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config)
    model_CKPT = torch.load(style_encoder_checkpoint_path, map_location="cpu")
    model_ckpt = {}
    for key, value in model_CKPT['model'].items():
        new_key = key[7:]
        model_ckpt[new_key] = value
    style_encoder.load_state_dict(model_ckpt)
    generator = JETSGenerator(conf).to(DEVICE)

    model_CKPT = torch.load(am_checkpoint_path, map_location=DEVICE)
    generator.load_state_dict(model_CKPT['generator'])
    generator.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    with open(config.token_list_path, 'r') as f:
        token2id = {t.strip():idx for idx, t, in enumerate(f.readlines())}

    with open(config.speaker2id_path, encoding='utf-8') as f:
        speaker2id = {t.strip():idx for idx, t in enumerate(f.readlines())}


    return (style_encoder, generator, tokenizer, token2id, speaker2id)

def get_style_embedding(prompt, tokenizer, style_encoder):
    prompt = tokenizer([prompt], return_tensors="pt")
    input_ids = prompt["input_ids"]
    token_type_ids = prompt["token_type_ids"]
    attention_mask = prompt["attention_mask"]
    with torch.no_grad():
        output = style_encoder(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )
    style_embedding = output["pooled_output"].cpu().squeeze().numpy()
    return style_embedding

def tts(name, text, prompt, content, speaker, models):
    (style_encoder, generator, tokenizer, token2id, speaker2id)=models
    

    style_embedding = get_style_embedding(prompt, tokenizer, style_encoder)
    content_embedding = get_style_embedding(content, tokenizer, style_encoder)

    speaker = speaker2id[speaker]

    text_int = [token2id[ph] for ph in text.split()]
    
    sequence = torch.from_numpy(np.array(text_int)).to(DEVICE).long().unsqueeze(0)
    sequence_len = torch.from_numpy(np.array([len(text_int)])).to(DEVICE)
    style_embedding = torch.from_numpy(style_embedding).to(DEVICE).unsqueeze(0)
    content_embedding = torch.from_numpy(content_embedding).to(DEVICE).unsqueeze(0)
    speaker = torch.from_numpy(np.array([speaker])).to(DEVICE)

    with torch.no_grad():

        infer_output = generator(
                inputs_ling=sequence,
                inputs_style_embedding=style_embedding,
                input_lengths=sequence_len,
                inputs_content_embedding=content_embedding,
                inputs_speaker=speaker,
                alpha=1.0
            )

    audio = infer_output["wav_predictions"].squeeze()* MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')

    return audio

speakers = config.speakers
models = get_models()




def new_line(i):
    col1, col2, col3, col4 = st.columns([1, 1, 3, 1])
    with col1:
        speaker=st.selectbox("说话人/speaker", speakers, key=f"{i}_speaker")
    with col2:
        prompt=st.text_input("提示/ prompt", "无", key=f"{i}_prompt")
    with col3:
        content=st.text_input("文本/text", "合成文本", key=f"{i}_text")
    
    with col4:
        lang=st.selectbox("语言/lang", ["ch", "us"], key=f"{i}_lang")
    

    

    flag = st.button(f"合成 / synthesize", key=f"{i}_button1")
    if flag:
        if lang=="us":
            if contains_chinese(content):
                st.info("文本含有中文/input texts contain chinese")
            else:
                text = g2p_en(content)
                path = tts(i, text, prompt, content, speaker, models)
                st.audio(path, sample_rate=config.sampling_rate)
        else:
            if not contains_chinese(content):
                st.info("文本含有英文/input texts contain english")
            else:            
                text = g2p_cn(content)
                path = tts(i, text, prompt, content, speaker, models)
                st.audio(path, sample_rate=config.sampling_rate)




new_line(0)
