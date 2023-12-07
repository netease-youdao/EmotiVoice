# Prediction interface for Cog ⚙️DEVICE
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import List

import numpy as np
from yacs import config as CONFIG
import torch
import re
import os, glob
import time
import subprocess
import requests
import soundfile as sf

from frontend_cn import g2p_cn
from frontend_en import preprocess_english
from config.joint.config import Config
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from transformers import AutoTokenizer

MAX_WAV_VALUE = 32768.0

# url for the weights mirror
REPLICATE_WEIGHTS_URL = "https://weights.replicate.delivery/default"

# files to download from the weights mirrors
DEFAULT_WEIGHTS = [
    {
        "dest": "outputs/prompt_tts_open_source_joint/ckpt",
        "src": "EmotiVoice",
        "files": [
            "do_00140000",
            "g_00140000",
        ],
    },
    {
        "dest": "outputs/style_encoder/ckpt",
        "src": "EmotiVoice",
        "files": [
            "checkpoint_163431",
        ],
    },
    {
        "dest": "WangZeJun/simbert-base-chinese",
        "src": "simbert-base-chinese/b5c82a8ab1e4bcac799620fc4d870aae087b0c71",
        "files": [
            "pytorch_model.bin",
            "config.json",
            "vocab.txt",
        ],
    }
]

def scan_checkpoint(cp_dir, prefix, c=8):
    pattern = os.path.join(cp_dir, prefix + '?'*c)
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def g2p_en(text):
    return preprocess_english(text)

def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = re.search(pattern, text)
    return match is not None

def download_json(url: str, dest: Path):
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")

def download_weights(baseurl: str, basedest: str, files: List[str]):
    """Download model weights from Replicate and save to file.
    Weights and download locations are specified in DEFAULT_WEIGHTS
    """
    basedest = Path(basedest)
    start = time.time()
    print("downloading to: ", basedest)
    basedest.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = basedest / f
        url = os.path.join(REPLICATE_WEIGHTS_URL, baseurl, f)
        if not dest.exists():
            print("downloading url: ", url)
            if dest.suffix == ".json":
                download_json(url, dest)
            else:
                subprocess.check_call(["pget", url, str(dest)], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):

    def setup_models(self):
        config = self.config
        am_checkpoint_path = scan_checkpoint(f'{config.output_directory}/prompt_tts_open_source_joint/ckpt', 'g_')

        style_encoder_checkpoint_path = scan_checkpoint(f'{config.output_directory}/style_encoder/ckpt', 'checkpoint_', 6)

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
        generator = JETSGenerator(conf).to(self.device)

        model_CKPT = torch.load(am_checkpoint_path, map_location=self.device)
        generator.load_state_dict(model_CKPT['generator'])
        generator.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

        with open(config.token_list_path, 'r') as f:
            self.token2id = {t.strip():idx for idx, t, in enumerate(f.readlines())}

        with open(config.speaker2id_path, encoding='utf-8') as f:
            self.speaker2id = {t.strip():idx for idx, t in enumerate(f.readlines())}

        self.style_encoder = style_encoder
        self.generator = generator
        print(self.tokenizer)

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        for weight in DEFAULT_WEIGHTS:
            download_weights(weight["src"], weight["dest"], weight["files"])
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_models()

    # def get_style_embedding(prompt, tokenizer, style_encoder):
    def get_style_embedding(self, text):
        tokenizer = self.tokenizer
        style_encoder = self.style_encoder
        text = tokenizer([text], return_tensors="pt")
        input_ids = text["input_ids"]
        token_type_ids = text["token_type_ids"]
        attention_mask = text["attention_mask"]
        with torch.no_grad():
            output = style_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        style_embedding = output["pooled_output"].cpu().squeeze().numpy()
        return style_embedding

    def tts(self, text, prompt, content, speaker):
        style_embedding = self.get_style_embedding(prompt)
        content_embedding = self.get_style_embedding(content)
        device = self.device

        speaker = self.speaker2id[speaker]

        text_int = [self.token2id[ph] for ph in text.split()]

        sequence = torch.from_numpy(np.array(text_int)).to(device).long().unsqueeze(0)
        sequence_len = torch.from_numpy(np.array([len(text_int)])).to(device)
        style_embedding = torch.from_numpy(style_embedding).to(device).unsqueeze(0)
        content_embedding = torch.from_numpy(content_embedding).to(device).unsqueeze(0)
        speaker = torch.from_numpy(np.array([speaker])).to(device)

        with torch.no_grad():

            infer_output = self.generator(
                    inputs_ling=sequence,
                    inputs_style_embedding=style_embedding,
                    input_lengths=sequence_len,
                    inputs_content_embedding=content_embedding,
                    inputs_speaker=speaker,
                    alpha=1.0
                )

        audio = infer_output["wav_predictions"].squeeze()* MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        path = os.path.join(self.config.output_directory,"output.mp3")
        sf.write(file=path, data=audio, samplerate=self.config.sampling_rate)
        return path

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Happy",
        ),
        content: str = Input(
            description="Input text",
            default="Emoti-Voice - a Multi-Voice and Prompt-Controlled T-T-S Engine",
        ),
        language: str = Input(
            description="Language",
            choices=["English", "Chinese"],
            default="English",
        ),
        speaker: str = Input(
            description="speakers",
            choices=Config().speakers,
            default=Config().speakers[0],
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        if language=="English":
            if contains_chinese(content):
                raise ValueError("文本含有中文/input text contains Chinese, but language is English")
            else:
                text = g2p_en(content)
                path = self.tts(text, prompt, content, speaker)
                return Path(path)
        else:
            if not contains_chinese(content):
                raise ValueError("文本含有英文/input text contains English, but language is Chinese")
            else:
                text = g2p_cn(content)
                path = self.tts(text, prompt, content, speaker)
                return Path(path)
