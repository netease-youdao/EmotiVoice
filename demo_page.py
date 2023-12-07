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

from frontend import g2p_cn_en, ROOT_DIR, read_lexicon, G2p
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
         
- Simply select a **Speaker ID**, type in the **text** you want to convert and the emotion **Prompt**, like a single word or even a sentence. Then click on the **Synthesize** button below to start voice synthesis.

- You can download the audio by clicking on the vertical three points next to the displayed audio widget.

- For more information on **'Speaker ID'**, please consult the [EmotiVoice voice wiki page](https://github.com/netease-youdao/EmotiVoice/tree/main/data/youdao/text)

- This interactive demo page is provided under the {html} file. The audio is synthesized by AI. 音频由AI合成，仅供参考。

""", unsafe_allow_html=True)

def scan_checkpoint(cp_dir, prefix, c=8):
    pattern = os.path.join(cp_dir, prefix + '?'*c)
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

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
    style_encoder.load_state_dict(model_ckpt, strict=False)
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
lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
g2p = G2p()

def new_line(i):
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 3.5, 1.3])
    with col1:
        speaker=st.selectbox("Speaker ID (说话人)", speakers, key=f"{i}_speaker")
    with col2:
        prompt=st.text_input("Prompt (开心/悲伤)", "", key=f"{i}_prompt")
    with col3:
        content=st.text_input("Text to be synthesized into speech (合成文本)", "合成文本", key=f"{i}_text")
    with col4:
        lang=st.selectbox("Language (语言)", ["zh_us"], key=f"{i}_lang")

    flag = st.button(f"Synthesize (合成)", key=f"{i}_button1")
    if flag:
        text =  g2p_cn_en(content, g2p, lexicon)
        path = tts(i, text, prompt, content, speaker, models)
        st.audio(path, sample_rate=config.sampling_rate)



new_line(0)
