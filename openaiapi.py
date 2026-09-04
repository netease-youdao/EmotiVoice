import logging
import os
import io
import torch
import glob

from fastapi import FastAPI, Response
from pydantic import BaseModel

from frontend import g2p_cn_en, ROOT_DIR, read_lexicon, G2p
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from transformers import AutoTokenizer
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
from pydub import AudioSegment
from yacs import config as CONFIG
from config.joint.config import Config
from speakers.main import router as speakers_router

LOGGER = logging.getLogger(__name__)

DEFAULTS = {
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
config = Config()
MAX_WAV_VALUE = 32768.0


def get_env(key):
    return os.environ.get(key, DEFAULTS.get(key))


def get_int_env(key):
    return int(get_env(key))


def get_float_env(key):
    return float(get_env(key))


def get_bool_env(key):
    return get_env(key).lower() == 'true'


def scan_checkpoint(cp_dir, prefix, c=8):
    pattern = os.path.join(cp_dir, prefix + '?'*c)
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def get_models():

    am_checkpoint_path = scan_checkpoint(
        f'{config.output_directory}/prompt_tts_open_source_joint/ckpt', 'g_')

    # f'{config.output_directory}/style_encoder/ckpt/checkpoint_163431'
    style_encoder_checkpoint_path = scan_checkpoint(
        f'{config.output_directory}/style_encoder/ckpt', 'checkpoint_', 6)

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
        token2id = {t.strip(): idx for idx, t, in enumerate(f.readlines())}

    with open(config.speaker2id_path, encoding='utf-8') as f:
        speaker2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}

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


def emotivoice_tts(text, prompt, content, speaker, models):
    (style_encoder, generator, tokenizer, token2id, speaker2id) = models

    style_embedding = get_style_embedding(prompt, tokenizer, style_encoder)
    content_embedding = get_style_embedding(content, tokenizer, style_encoder)

    speaker = speaker2id[speaker]

    text_int = [token2id[ph] for ph in text.split()]

    sequence = torch.from_numpy(np.array(text_int)).to(
        DEVICE).long().unsqueeze(0)
    sequence_len = torch.from_numpy(np.array([len(text_int)])).to(DEVICE)
    style_embedding = torch.from_numpy(style_embedding).to(DEVICE).unsqueeze(0)
    content_embedding = torch.from_numpy(
        content_embedding).to(DEVICE).unsqueeze(0)
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

    audio = infer_output["wav_predictions"].squeeze() * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')

    return audio


speakers = config.speakers
models = get_models()
app = FastAPI()
app.include_router(speakers_router)
lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
g2p = G2p()

from typing import Optional
class SpeechRequest(BaseModel):
    input: str
    voice: str = '8051'
    prompt: Optional[str] = ''
    language: Optional[str] = 'zh_us'
    model: Optional[str] = 'emoti-voice'
    response_format: Optional[str] = 'mp3'
    speed: Optional[float] = 1.0


@app.post("/v1/audio/speech")
def text_to_speech(speechRequest: SpeechRequest):

    text = g2p_cn_en(speechRequest.input, g2p, lexicon)
    np_audio = emotivoice_tts(text, speechRequest.prompt,
                              speechRequest.input, speechRequest.voice,
                              models)
    y_stretch = np_audio
    if speechRequest.speed != 1.0:
        y_stretch = pyrb.time_stretch(np_audio, config.sampling_rate, speechRequest.speed)
    wav_buffer = io.BytesIO()
    sf.write(file=wav_buffer, data=y_stretch,
             samplerate=config.sampling_rate, format='WAV')
    buffer = wav_buffer
    response_format = speechRequest.response_format
    if response_format != 'wav':
        wav_audio = AudioSegment.from_wav(wav_buffer)
        wav_audio.frame_rate=config.sampling_rate
        buffer = io.BytesIO()
        wav_audio.export(buffer, format=response_format)

    return Response(content=buffer.getvalue(),
                    media_type=f"audio/{response_format}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)