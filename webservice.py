import os, glob
import numpy as np
from yacs import config as CONFIG
import torch
import soundfile as sf
from flask import Flask, jsonify, request, send_file
import io

from frontend import g2p_cn_en, ROOT_DIR, read_lexicon, G2p
from config.joint.config import Config
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from transformers import AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WAV_VALUE = 32768.0

config = Config()

def scan_checkpoint(cp_dir, prefix, c=8):
    pattern = os.path.join(cp_dir, prefix + '?'*c)
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

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

def tts(text, prompt, content, speaker, models):
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

import nltk
nltk.download('averaged_perceptron_tagger_eng')

speakers = config.speakers
models = get_models()
lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
g2p = G2p()

app = Flask(__name__)

@app.route('/speakers', methods=['GET'])
def get_speakers():
    return jsonify(speakers)

@app.route('/generate', methods=['POST'])
def generate_speech():
    data = request.get_json()
    content = data.get('content')
    prompt = data.get('prompt')
    speaker = data.get('speaker')
    
    if not all([content, prompt, speaker]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    text = g2p_cn_en(content, g2p, lexicon)
    audio = tts(text, prompt, content, speaker, models)
    
    # Convert audio to bytes
    byte_io = io.BytesIO()
    sf.write(byte_io, audio, 22050, format='WAV')
    byte_io.seek(0)
    
    return send_file(
        byte_io,
        mimetype='audio/wav',
        as_attachment=True,
        download_name='generated_speech.wav'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
